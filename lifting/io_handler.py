from clang.cindex import Index, CursorKind
import itertools
import json
from typing import Dict, List, NamedTuple

LOOP_TYPES = [CursorKind.FOR_STMT, CursorKind.WHILE_STMT, CursorKind.DO_STMT]


class Variable(NamedTuple):
    """
    A Variable object holds its dimension (or rank) and a list of integer values.

    Attributes:
        dimension (int): The dimension of the variable (1 for scalars,
                         or e.g. length for a 1D array).
        values (List[int]): The actual numeric data for this variable.
    """

    dimension: int
    values: List[int]


class IOExample(NamedTuple):
    """
    An IOExample encapsulates a single I/O sample (test case), including:
      - input: A dictionary mapping input variable names to `Variable` objects.
      - constants: A dictionary mapping symbolic names (e.g. 'Cons1') to their integer values.
      - output: A single `Variable` object representing the program output for this sample.

    This format makes it easy to pass I/O samples around and test candidate programs.
    """

    input: Dict[str, Variable]
    constants: Dict[str, int]
    output: Variable


def parse_program(program_path):
    """
    Parse a C program using clang's Python bindings and produce a translation unit (TU).
    The TU can then be traversed to extract AST information.

    Args:
        program_path (str): Path to the .c file to parse.

    Returns:
        clang.cindex.TranslationUnit: The resulting translation unit.
    """
    idx = Index.create()
    tu = idx.parse(program_path, args=["-c"])
    return tu


def extract_clang(cursor):
    """
    Extract the raw source text for the given AST cursor's extent.

    Args:
        cursor (clang.cindex.Cursor): The AST cursor.

    Returns:
        str: The substring of the file corresponding to the cursor's start->end range.

    Note:
        This function opens the cursor's file, reads it in full, and then slices
        according to the cursor's start and end offsets.
    """
    if cursor is None:
        return ""
    filename = cursor.location.file.name
    with open(filename, "r") as fh:
        contents = fh.read()
    return contents[cursor.extent.start.offset : cursor.extent.end.offset]


def get_nodes_by_kind(tu, kinds):
    """
    Walk the AST of the given translation unit and collect nodes of certain cursor kinds.

    Args:
        tu (clang.cindex.TranslationUnit): The parsed translation unit.
        kinds (list of clang.cindex.CursorKind): A list of cursor kinds to filter on.

    Returns:
        list of clang.cindex.Cursor: All nodes whose kind is in 'kinds'.
    """
    return [n for n in tu.cursor.walk_preorder() if n.kind in kinds]


def get_loop_control_vars(tu):
    """
    Identify the variables used as loop control in for, while, and do statements.

    For classical 'for' loops, for instance, the loop condition often appears as the
    second child in the AST. This function attempts to find the variable used there.

    Args:
        tu (clang.cindex.TranslationUnit): The parsed translation unit.

    Returns:
        set of int: A set of hashes for the AST definitions of loop control variables.
                    This can be used to detect if a variable is purely used to control loops.
    """
    loops = get_nodes_by_kind(tu, LOOP_TYPES)
    loop_control_vars = set()
    for l in loops:
        # We assume that the loops is the classical form, therefore, the loop condition
        # is the second element in the list formed by the children of the loop node in the AST.
        # Variables are saved using their hash to avoid duplicates.
        loop_cond = list(l.get_children())[1]
        if loop_cond.kind == CursorKind.BINARY_OPERATOR:
            loop_control_vars.add(
                list(loop_cond.get_children())[0].get_definition().hash
            )

    return loop_control_vars


def get_assignments(tu):
    """
    Find all assignment expressions (including compound assignments like +=, -=) in the code.

    Args:
        tu (clang.cindex.TranslationUnit): The parsed translation unit.

    Returns:
        list of clang.cindex.Cursor: The cursor objects representing these assignments.
    """
    binop_exprs = get_nodes_by_kind(tu, [CursorKind.BINARY_OPERATOR])
    compound_assigments = get_nodes_by_kind(
        tu, [CursorKind.COMPOUND_ASSIGNMENT_OPERATOR]
    )
    return compound_assigments + [
        binop
        for binop in binop_exprs
        for tok in binop.get_tokens()
        if "=" == tok.spelling
    ]


def get_constants(program_path):
    """
    Analyze a C program to find integer literals that appear on the RHS of assignment statements
    (excluding loop control variable initializations).

    Args:
        program_path (str): Path to the .c file to parse.

    Returns:
        set of int: Distinct integer constants found in the relevant positions.
    """
    tu = parse_program(program_path)
    assignments = get_assignments(tu)
    cons = set()
    visited = set()
    loop_vars = get_loop_control_vars(tu)
    for a in assignments:
        lhs = list(a.get_children())[0]
        if lhs.kind == CursorKind.DECL_REF_EXPR:
            # We do not consider constants used to initialize loop variables.
            if lhs.get_definition().hash in loop_vars:
                continue

        # We keep track of the visit constant nodes to avoid duplicates.
        for c in list(a.get_children())[1].walk_preorder():
            # We are only interested in constants that appear on the RHS of assignments.
            if c.kind == CursorKind.UNARY_OPERATOR:
                unary_operand = list(c.get_children())[0]
                if (
                    unary_operand.kind == CursorKind.INTEGER_LITERAL
                    and unary_operand.hash not in visited
                ):
                    cons.add(int(extract_clang(c)))
                    visited.add(unary_operand.hash)

            elif c.kind == CursorKind.INTEGER_LITERAL and c.hash not in visited:
                cons.add(int(extract_clang(c)))

            visited.add(c.hash)

    return cons


class IOHandler:
    """
    IOHandler processes a JSON file containing I/O samples, merging any 
    discovered constants in the program with the data from the JSON. 
    The result is a list of IOExample objects that can be used to 
    test candidate programs or transformations.
    """


    @staticmethod
    def parse_io(program_path, io_path):
        """
        Parse a JSON file of I/O samples and gather relevant constants from
        the corresponding C program. Create a list of IOExample objects 
        containing input, constants, and output.

        Args:
            program_path (str): Path to the C source file for analyzing constants.
            io_path (str): Path to the JSON file containing I/O samples.

        Returns:
            list of IOExample: Each IOExample corresponds to one sample in the JSON file.

        Raises:
            FileNotFoundError: If the JSON file is not found.
        """
        io_set = []
        try:
            with open(io_path, "r") as io_file:
                io_pairs = json.load(io_file)
        except FileNotFoundError as e:
            raise e

        # The IO files do not hold information regarding constants in the original program. We analyze the
        # source code to retrieve relevant constants. In case no constants are found, the Constants field
        # in the IOExample object will be an empty dictionary.
        io_constants = dict()
        constants = get_constants(program_path)
        if constants:
            constant_id = itertools.count(1)
            for c in constants:
                io_constants[f"Cons{next(constant_id)}"] = c

        for sample in io_pairs:
            io_input_vars = dict()

            output_values = list(sample["output"].values())[0]
            io_output_var = Variable(
                int(output_values[0]),
                [int(value) for value in output_values[1].split()],
            )

            for in_var, in_values in sample["input"].items():
                io_input_vars[in_var] = Variable(
                    int(in_values[0]), [int(val) for val in in_values[1].split()]
                )

            io_set.append(IOExample(io_input_vars, io_constants, io_output_var))

        return io_set

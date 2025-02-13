import re


def _get_tensor_name(tensor):
    """
    Extract the 'base' name of the tensor from its string representation, ignoring
    possible leading '-' (negation) and the 'Cons' prefix if present.

    For example:
        - 'Cons(x)' -> 'Cons(x)' is a constant placeholder.
        - '-b(i)' -> 'b'
        - 'c(i,j)' -> 'c'
    
    Args:
        tensor (str): The string form of the tensor, which may include a
                      negation prefix '-' or start with 'Cons'.

    Returns:
        str: The base name (e.g., 'b', 'c', or 'Cons(x)').
    """
    if tensor.startswith("Cons"):
        return tensor[1:] if tensor[0] == "-" else tensor
    else:
        return tensor[1] if tensor[0] == "-" else tensor[0]


def _get_tensor_order(tensor):
    """
    Determine the order (dimension) of the tensor by counting the commas in its 
    indexing parentheses.

    Args:
        tensor (str): The string representation of a tensor (e.g., 'b(i,j)', 'c(i)', etc.)

    Returns:
        int: The order of the tensor. For example:
             - 'b(i,j)' -> 2
             - 'x(i)' -> 1
             - 'z' (no parentheses) -> 0
    """
    return 0 if tensor.count("(") == 0 else tensor.count(",") + 1


def _get_tensor_indexing(tensor):
    """
    Extract the substring representing the indices of a tensor, ignoring negation
    and special 'Cons' patterns.

    For example:
        - 'b(i,j)' -> 'i,j'
        - '-b(i)' -> 'i'
        - 'Cons(x)' -> ''
        - 'a' -> ''

    Args:
        tensor (str): The full tensor string, possibly negated or containing 'Cons'.

    Returns:
        str: The indices (e.g., 'i,j') as a comma-separated string, or an empty string if none.
    """
    if len(tensor) == 1 or re.match("Cons*.", tensor):
        return ""
    elif tensor[0] == "-":
        return tensor[2:].lstrip("(").rstrip(")")
    else:
        return tensor[1:].lstrip("(").rstrip(")")


def _is_negated(tensor):
    """
    Check if the given tensor string has a leading '-' character indicating negation.

    Args:
        tensor (str): The tensor string, e.g., '-b(i,j)'.

    Returns:
        bool: True if negated, False otherwise.
    """
    return True if tensor[0] == "-" else False


class Tensor:
    """
    Represents a tensor in the TACO IR with information about:
      - Name (e.g. 'b')
      - Order (dimension)
      - Indexing (string form of indices, e.g. 'i,j')
      - Negation (boolean)

    Attributes:
        name (str): The base name of the tensor (e.g. 'b').
        order (int): The number of indices (e.g., 2 for b(i,j)).
        indexing (str): A comma-separated string of index names.
        negated (bool): Whether the tensor is negated (e.g., '-b(i)').
    """

    def __init__(self, name, order, indexing="", negated=False):
        """
        Initialize a Tensor instance.

        Args:
            name (str): The tensor's base name.
            order (int): The number of indices (dimension).
            indexing (str): A string of comma-separated index variables.
            negated (bool): True if the tensor is negated.
        """
        self.name = name
        self.order = order
        self.indexing = indexing
        self.negated = negated

    def __str__(self):
        """
        Return a string representing the tensor. If negated, prepend '-'.
        If order > 0, append ':' followed by indexing (i.e., 'b:i,j').

        Returns:
            str: The string form of the tensor.
        """
        tensor_as_str = "-" + self.name if self.negated else self.name
        tensor_as_str = (
            tensor_as_str + ":" + self.indexing if self.order > 0 else tensor_as_str
        )
        return tensor_as_str

    def __eq__(self, other):
        """
        Compare this tensor to another by name.

        Returns:
            bool: True if names match, False otherwise.
        """
        if isinstance(other, Tensor):
            return self.name == self.name

    def __hash__(self):
        """
        Hash the tensor based on its name (to allow usage in sets/dicts).

        Returns:
            int: Hash of the tensor's name.
        """
        return hash(self.name)

    def copy(self):
        """
        Create a new Tensor with identical attributes.

        Returns:
            Tensor: A deep copy of this Tensor.
        """
        return Tensor(self.name, self.order, self.indexing, self.negated)


class TACOProgram:
    """
    Represents a TACO program of the form:

        <lhs_tensor> = <rhs_tensor_0> <op_0> <rhs_tensor_1> <op_1> ... <rhs_tensor_n>

    It tracks:
      - The LHS tensor,
      - A list of RHS tensors,
      - A list of operators,
      - Parentheses positions in the RHS.

    Attributes:
        lhs_tensor (Tensor): The left-hand side tensor in the assignment.
        rhs_tensors (list of Tensor): The list of tensors on the right-hand side.
        arithops (list of str): The binary arithmetic operators ('+', '-', '*', '/').
        n_tensors (int): The total number of tensors (LHS + RHS).
        n_arithops (int): The total number of arithmetic operators in the expression.
        l_par (dict): Mapping from tensor index to left parenthesis strings.
        r_par (dict): Mapping from tensor index to right parenthesis strings.
    """

    def __init__(self, lhs_tensor, rhs_tensors=[], arithops=[], l_par=None, r_par=None):
        """
        Initialize a TACOProgram object.

        Args:
            lhs_tensor (Tensor): The tensor on the LHS of the assignment.
            rhs_tensors (list of Tensor): Tensors appearing on the RHS.
            arithops (list of str): Operators interleaved among RHS tensors (binary ops).
            l_par (dict): Left parenthesis positions (index -> string).
            r_par (dict): Right parenthesis positions (index -> string).

        Raises:
            AssertionError: If rhs_tensors is empty or the number of operators 
                            does not match the expected pattern (#tensors = #ops + 1).
        """
        assert (
            rhs_tensors
        ), "Program should have at least one tensors on the right-hand side."
        if arithops:
            assert (
                len(rhs_tensors) == len(arithops) + 1
            ), "The number of tensors should be the number of binary operators plus one."

        self.lhs_tensor = lhs_tensor
        self.rhs_tensors = rhs_tensors
        self.tensor_names = [lhs_tensor.name] + [
            tensor.name for tensor in self.rhs_tensors
        ]
        self.indexings = [lhs_tensor.indexing] + [
            tensor.indexing for tensor in self.rhs_tensors
        ]
        self.arithops = arithops
        self.n_tensors = 1 + len(rhs_tensors)
        self.n_arithops = self.n_tensors - 2
        self.l_par = l_par
        self.r_par = r_par

    def __str__(self):
        """
        Build a string representation of the TACO program with optional parentheses 
        around certain RHS tensors. Format:

            lhs_tensor = (rhs_tensor0 op0 rhs_tensor1) op1 (rhs_tensor2)

        Returns:
            str: The textual representation of this TACO program.
        """
        program_as_str = str(self.lhs_tensor) + " = "
        if self.n_tensors == 2:
            if 0 in self.l_par:
                return (
                    program_as_str
                    + self.l_par[0]
                    + str(self.rhs_tensors[0])
                    + self.r_par[0]
                )
            else:
                return program_as_str + str(self.rhs_tensors[0])
        else:
            for i in range(self.n_arithops):
                if i in self.l_par:
                    program_as_str += self.l_par[i]
                program_as_str += str(self.rhs_tensors[i])
                if i in self.r_par:
                    program_as_str += self.r_par[i]
                program_as_str += " " + str(self.arithops[i]) + " "

            program_as_str += str(self.rhs_tensors[self.n_tensors - 2])
            if self.n_tensors - 2 in self.r_par:
                program_as_str += self.r_par[self.n_tensors - 2]

            return program_as_str

    def __repr__(self):
        """
        Return the same representation as __str__ for debugging convenience.
        """
        return self.__str__()

    def copy(self):
        """
        Create a deep copy of the TACOProgram, duplicating lhs, rhs, and operators.

        Returns:
            TACOProgram: A new instance with the same structure.
        """
        lhs = self.lhs_tensor.copy()
        rhs = [t.copy() for t in self.rhs_tensors]
        return TACOProgram(lhs, rhs, self.arithops.copy(), self.mutation)

    def has_variable_tensor(self):
        """
        Check if any RHS tensors are non-numeric (i.e., not purely numeric constants).

        Returns:
            bool: True if at least one tensor is not numeric, False otherwise.
        """
        return any(not tensor.name.isnumeric() for tensor in self.rhs_tensors)

    def get_reference_to_tensor(self, tensor_name):
        """
        Retrieve a reference to the first RHS tensor with the specified name.

        Args:
            tensor_name (str): The name of the tensor to find among rhs_tensors.

        Returns:
            Tensor or None: The matching Tensor if found, otherwise None.
        """
        for tensor in self.rhs_tensors:
            if tensor.name == tensor_name:
                return tensor

    def has_tensor_negated(self):
        """
        Check if any RHS tensor is marked as negated.

        Returns:
            bool: True if at least one RHS tensor has negated == True, otherwise False.
        """
        return any(tensor.negated for tensor in self.rhs_tensors)

    @classmethod
    def from_string(cls, program):
        """
        Build a TACOProgram from a string representation of the form:
            "lhs_tensor = rhs_expr"

        The right-hand side expression can contain:
            - Parentheses '(' and ')'
            - Tensors (possibly negated), e.g. -b(i), c(i,j), b
            - Arithmetic operators (+, -, *, /)
        Parentheses are stored in dictionaries 'l_par' and 'r_par', keyed by the 
        index of the tensor they wrap around.

        Args:
            program (str): A string like "a(i) = (b(i) + c(i))".

        Returns:
            TACOProgram: A new TACOProgram object parsed from the string.
        """
        tensors_regex = "^-?[b-z]"
        cons_placeholder_regex = "-?Cons*"
        tensors = []
        arithops = []
        l_par = dict()
        r_par = dict()

        lhs, rhs = program.split(" = ")
        lhs_tensor = Tensor(
            _get_tensor_name(lhs), _get_tensor_order(lhs), _get_tensor_indexing(lhs)
        )

        rhs_pattern = r"(\(|\)|(?<!\w)-?[a-zA-Z_]\w*\([a-zA-Z_,\d]*\)|(?<!\w)-?[a-zA-Z_]\w*|(?<!\w)[+\*/-](?![a-zA-Z_]))"
        rhs_tensor_idx = 0
        for elem in re.findall(rhs_pattern, rhs):
            if re.match(tensors_regex, elem) or re.match(cons_placeholder_regex, elem):
                tensors.append(
                    Tensor(
                        _get_tensor_name(elem),
                        _get_tensor_order(elem),
                        _get_tensor_indexing(elem),
                        _is_negated(elem),
                    )
                )
            elif elem == "(":
                if rhs_tensor_idx in l_par:
                    l_par[rhs_tensor_idx] += elem
                else:
                    l_par[rhs_tensor_idx] = elem
            elif elem == ")":
                if rhs_tensor_idx in r_par:
                    r_par[rhs_tensor_idx] += elem
                else:
                    r_par[rhs_tensor_idx] = elem
            else:
                arithops.append(elem)
                rhs_tensor_idx += 1

        return cls(lhs_tensor, tensors, arithops, l_par, r_par)

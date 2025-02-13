

import re
import ast

# ------------------------------------------------------------------------------
# Data Classes
# ------------------------------------------------------------------------------
class Tensor:
    """
    Represents a tensor that may have one or more indices (i.e., dimensions).
    Includes a negated flag to indicate unary-minus usage.

    Attributes:
        name (str): The short name of the tensor (e.g. 'a', 'b', 'c').
        indices (list of str): A list of index names, e.g. ['i', 'j'].
        negated (bool): True if the tensor has been negated (e.g. '-b(i,j)').
    """

    def __init__(self, name, indices, negated=False):
        """
        Constructor for the Tensor class.

        Args:
            name (str): The string name for this tensor, e.g. 'b'.
            indices (list of str): The index variables, e.g. ['i', 'j'].
            negated (bool): Flag indicating if this tensor is negated.
        """
        self.name = name
        self.indices = indices
        self.negated = negated  # Track if the tensor is negated

    def __repr__(self):
        negation = '-' if self.negated else ''
        return f"{negation}{self.name}({','.join(self.indices)})"

class Constant:
    """
    Represents a numeric constant within an expression.

    Attributes:
        name (str): The string representation of the constant, e.g. '3'.
    """
    def __init__(self, name):
        """
        Constructor for Constant class.

        Args:
            name (str): The literal string value of the constant (e.g., '3' or '-4').
        """
        self.name = name

    def __repr__(self):
        return self.name

class Operation:
    """
    Represents a binary operation between two subexpressions.

    Attributes:
        left: The left subexpression (could be Tensor, Constant, or nested Operation).
        operator (str): A string representing the operator, e.g. '+', '-', '*', '/'.
        right: The right subexpression.
    """

    def __init__(self, left, operator, right):
        """
        Constructor for Operation class.

        Args:
            left: Left-hand expression (Tensor, Constant, or Operation).
            operator (str): The operator symbol, e.g. '+', '*', etc.
            right: Right-hand expression (Tensor, Constant, or Operation).
        """
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.operator} {self.right})"

class Assignment:
    """
    Represents an assignment statement in a tensor algebraic expression.

    Attributes:
        lhs (Tensor): The left-hand-side tensor to be assigned.
        rhs (Operation, Tensor, or Constant): The right-hand-side expression.
    """

    def __init__(self, lhs, rhs):
        """
        Constructor for Assignment class.

        Args:
            lhs (Tensor): The tensor on the left-hand side of the assignment.
            rhs (Operation or Tensor or Constant): The expression on the right-hand side.
        """
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs} = {self.rhs}"




# ------------------------------------------------------------------------------
# GPTTensorParser Class
# ------------------------------------------------------------------------------
class GPTTensorParser:
    """
    A parser that reads lines describing tensor assignments and converts them
    into an internal representation using Tensor, Constant, Operation, and
    Assignment objects.

    Attributes:
        index_order (list): The valid list of index labels in order, e.g. ['i','j','k','l'].
        max_indices (int): The maximum number of distinct index variables allowed in an expression.
        current_index (int): Tracks how many unique indices have been used so far.
        index_map (dict): Maps encountered index variables (e.g. 'f') to standard ones (e.g. 'i','j','k').
        tensor_map (dict): Maps encountered tensor names (e.g. 'Mat1') to single-letter names (e.g. 'b','c','d').
        lhs_tensor (str): The single-letter tensor name for all LHS usage, defaults to 'a'.
    """

    def __init__(self):
        """
        Constructor for GPTTensorParser, initializing internal state 
        for index mapping and tensor naming.
        """
        self.index_order = ["i", "j", "k", "l"]
        self.max_indices = 4  # Maximum distinct indices allowed
        self.current_index = 0  # Tracks the current index being used
        self.index_map = {}  # Maps tensor indices (i.e., i, j) to 'i', 'j', etc.
        self.tensor_map = {}  # Maps tensor names (A, B) to 'b', 'c', 'd', etc. for RHS
        self.lhs_tensor = 'a'  # Always map LHS tensor to 'a'

    def reset(self):
        """
        Reset all mappings and counters for fresh processing of a new line.
        Ensures that each line's assignment is parsed independently.
        """
        self.current_index = 0
        self.index_map = {}
        self.tensor_map = {}
        self.lhs_tensor = 'a'  # Reset the LHS tensor to always map to 'a'

    def get_next_index(self):
        """
        Retrieve the next available index from self.index_order. 
        Raises an error if more than self.max_indices unique indices are used.

        Returns:
            str: The next index symbol, e.g. 'i', 'j', 'k', or 'l'.

        Raises:
            IndexError: If the number of distinct indices exceeds max_indices.
        """
        if self.current_index >= self.max_indices:
            raise IndexError("More than four distinct indices used.")
        index = self.index_order[self.current_index]
        self.current_index += 1
        return index

    def parse_tensor(self, expr, is_lhs=False, negated=False):
        """
        Parse a string that represents a tensor reference, e.g., "A(i,j)" or "B" 
        (with optional negation).

        Args:
            expr (str): The substring representing a tensor or constant, e.g. "A(i,j)" or "5".
            is_lhs (bool): Indicates if this tensor is on the left-hand side of an assignment.
            negated (bool): If True, the tensor is negated (e.g. '-b(i)').

        Returns:
            Tensor or Constant: A Tensor object if indices are found, otherwise a Constant.

        Raises:
            ValueError: If the string doesn't match a valid tensor or constant pattern.
        """
        match = re.match(r'([A-Za-z0-9]+)\(([^)]*)\)', expr)
        if match:
            name, indices = match.groups()
            indices = [index.strip() for index in indices.split(",")]

            if is_lhs:
                tensor_name = self.lhs_tensor
            else:
                # Map RHS tensor names to 'b', 'c', 'd', etc.
                if name not in self.tensor_map:
                    self.tensor_map[name] = chr(ord('b') + len(self.tensor_map))
                tensor_name = self.tensor_map[name]

            # Map indices (i, j, k, etc.) to 'i', 'j', 'k'
            for idx in indices:
                if idx not in self.index_map:
                    self.index_map[idx] = self.get_next_index()
            mapped_indices = [self.index_map[idx] for idx in indices]
            return Tensor(tensor_name, mapped_indices, negated=negated)
        elif re.match(r'[A-Za-z0-9]+', expr):  # It's a constant
            return Constant(expr)
        else:
            raise ValueError(f"Invalid tensor expression: {expr}")

    def parse_expression(self, expr):
        """
        Parse a right-hand-side (RHS) expression string using Python's ast module 
        to manage operator precedence and parentheses.

        Args:
            expr (str): The expression, e.g. "B(i) * (C(i) + 1)".

        Returns:
            (Operation, Tensor, or Constant): An internal representation of the expression tree.

        Raises:
            ValueError: If the expression is invalid or cannot be parsed.
        """
        expr = expr.strip()
        try:
            expr_ast = ast.parse(expr, mode='eval').body
            return self._parse_ast(expr_ast)
        except Exception as e:
            raise ValueError(f"Invalid expression: {expr}\nError: {e}")

    def _parse_ast(self, node):
        """
        Recursively walk an AST node and convert it into our internal representation 
        (Operation, Tensor, or Constant objects).

        Args:
            node (ast.AST): The AST node to process.

        Returns:
            Operation, Tensor, or Constant: The constructed representation.

        Raises:
            ValueError: If an unsupported AST structure or operator is encountered.
        """
        if isinstance(node, ast.BinOp):
            left = self._parse_ast(node.left)
            right = self._parse_ast(node.right)
            operator = self._get_operator(node.op)
            return Operation(left, operator, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._parse_ast(node.operand)
            if isinstance(node.op, ast.USub):
                if isinstance(operand, Constant):
                    # Negate the constant value
                    return Constant(f"-{operand.name}")
                elif isinstance(operand, Tensor):
                    operand.negated = not operand.negated
                    return operand
                else:
                    return Operation(Constant("0"), '-', operand)
            else:
                raise ValueError("Unsupported unary operator")
        elif isinstance(node, ast.Call):
            # Handle tensor calls like A(i)
            func_name = node.func.id
            indices = []
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    indices.append(arg.id)
                elif isinstance(arg, ast.Constant):
                    indices.append(str(arg.value))
                else:
                    raise ValueError(f"Unsupported index in tensor: {arg}")
            return self.parse_tensor(f"{func_name}({','.join(indices)})")
        elif isinstance(node, ast.Name):
            # Handle constants or tensors without indices
            return self.parse_tensor(node.id)
        elif isinstance(node, ast.Constant):
            # Handle numeric constants
            return Constant(str(node.value))
        else:
            raise ValueError(f"Unsupported AST node: {node}")

    def _get_operator(self, op):
        """
        Translate an ast operator object to our string-based operator representation.

        Args:
            op (ast.operator): The operator node from the AST (e.g. ast.Add()).

        Returns:
            str: Operator string ('+', '-', '*', '/').

        Raises:
            ValueError: If the operator is not recognized.
        """
        if isinstance(op, ast.Add):
            return '+'
        elif isinstance(op, ast.Sub):
            return '-'
        elif isinstance(op, ast.Mult):
            return '*'
        elif isinstance(op, ast.Div):
            return '/'
        else:
            raise ValueError(f"Unsupported operator: {op}")

    def parse_assignment(self, line):
        """
        Parse a full assignment line of the form: LHS = RHS.
        If ':=' is used instead of '=', replace it with '='.

        Args:
            line (str): A line containing an assignment, e.g. "A(i) = B(i) + 1".

        Returns:
            Assignment: The constructed assignment object.
        """
        if ":=" in line:
            line = line.replace(":=", "=")
        lhs, rhs = map(str.strip, line.split('='))
        lhs_tensor = self.parse_tensor(lhs, is_lhs=True)  # LHS is always 'a'
        rhs_expr = self.parse_expression(rhs)  # RHS tensors start from 'b'
        return Assignment(lhs_tensor, rhs_expr)

    def process_lines(self, lines):
        """
        Parse and transform a list of assignment strings into AST objects.

        Args:
            lines (list of str): Lines of expressions/assignments to parse.

        Returns:
            list of Assignment: A list of successfully parsed assignment objects.
        """
        asts = []
        for line in lines:
            self.reset()  # Reset mappings for each new line
            try:
                ast = self.parse_assignment(line)
                self.check_index_count(ast)
                asts.append(ast)
            except Exception:
                continue
        return asts

    def check_index_count(self, assignment):
        """
        Verify that the total set of unique indices used in the Assignment 
        does not exceed self.max_indices.

        Args:
            assignment (Assignment): The assignment object to check.

        Raises:
            IndexError: If more than self.max_indices distinct indices are found.
        """
        indices = set(assignment.lhs.indices)
        if isinstance(assignment.rhs, Operation):
            indices.update(self.collect_indices(assignment.rhs))
        elif isinstance(assignment.rhs, Tensor):
            indices.update(assignment.rhs.indices)
        if len(indices) > self.max_indices:
            raise IndexError(f"More than {self.max_indices} distinct indices used.")

    def collect_indices(self, expr):
        """
        Recursively gather all indices used in an expression.

        Args:
            expr (Operation or Tensor): The expression or subexpression.

        Returns:
            list of str: All indices that appear in the expression.
        """
        if isinstance(expr, Tensor):
            return expr.indices
        elif isinstance(expr, Operation):
            return self.collect_indices(expr.left) + self.collect_indices(expr.right)
        return []

    def generate_code(self, asts):
        """
        Produce a list of string representations from a list of Assignment objects.

        Args:
            asts (list of Assignment): List of assignment objects to serialize.

        Returns:
            list of str: String representations of each assignment.
        """
        code = []
        for ast in asts:
            code.append(str(ast))
        return code


# ------------------------------------------------------------------------------
# TensorRenamer Class
# ------------------------------------------------------------------------------
class TensorRenamer:
    """
    Renames multi-letter tensors in expressions to single-letter names 
    for brevity, starting from 'a' and incrementing.

    Attributes:
        tensor_map (dict): Maps original tensor names to new single-letter names.
        next_tensor_letter (str): The next available letter for renaming.
    """

    def __init__(self):
        """
        Constructor for TensorRenamer, initializing the mapping and first letter.
        """
        # This dictionary will map original tensor names (e.g., "arr") to single-letter names (e.g., "a")
        self.tensor_map = {}
        self.next_tensor_letter = 'a'

    def reset(self):
        """
        Reset the renamer to its initial state, clearing all mappings 
        and reverting the next available letter to 'a'.
        """
        self.tensor_map = {}
        self.next_tensor_letter = 'a'

    def rename_tensors(self, expr):
        """
        Rename all tensors in the input expression to single-letter tensor names
        from 'a' onwards. Tensors must match the pattern alphanumeric + parentheses.

        Args:
            expr (str): The expression string containing tensors to be renamed.

        Returns:
            str: The expression with renamed tensors.
        """
        # Regex to match tensor-like patterns (alphanumeric name followed by parentheses)
        tensor_pattern = re.compile(r'([A-Za-z0-9]+)\([A-Za-z0-9, ]*\)')

        # Function to replace tensor names with single-letter names
        def replace_tensor(match):
            original_tensor = match.group(1)  # Extract the tensor name (e.g., "arr")
            if original_tensor not in self.tensor_map:
                # If the tensor hasn't been renamed yet, assign it a new single-letter name
                self.tensor_map[original_tensor] = self.next_tensor_letter
                self.next_tensor_letter = chr(ord(self.next_tensor_letter) + 1)  # Move to the next letter
            renamed_tensor = self.tensor_map[original_tensor]  # Get the renamed tensor
            return match.group(0).replace(original_tensor, renamed_tensor)  # Replace tensor in the match

        # Use regex sub to replace all tensor occurrences in the expression
        renamed_expr = tensor_pattern.sub(replace_tensor, expr)
        return renamed_expr

# Example usage:

if __name__ == "__main__":
    parser = GPTTensorParser()
    lines = [
"t(i, k) = m1(i, j) * m2(j, k)",
"Target(i,k) = Mat1(i,f)*Mat2(f,k)",
"t(i, k) := m1(i, f) * (m2(k, f) + 1)",
"target(i, k) = sum(j, mat1(i, j) * mat2(j, k))"]
    asts = parser.process_lines(lines)
    code = parser.generate_code(asts)
    print('\n'.join(code))

import itertools
import re
from check import get_tensor_order

class Candidate():
    """
    Represents a candidate TACO program during synthesis. A Candidate 
    has a left-hand side (LHS), a right-hand side (RHS), a map of each 
    tensor to its corresponding order (number of indices), and a unique 
    numeric identifier.

    Attributes:
        id (int): A globally unique identifier (automatically assigned in sequence).
        lhs (str): The left-hand side of the assignment (e.g. 'A(i,j)').
        rhs (str): The right-hand side of the assignment (e.g. 'B(i,j) + C(i,j)').
        tensor_orders (dict): A dictionary mapping tensor names to their orders (dimensionalities).
    """

    id = itertools.count()
    tensors_regex = r'[-]?[a-zA-Z]+(?:\([^()]*\))?|Cons'
    constant_tensor_regex = r'Cons*.'

    def __init__(self, lhs, rhs):
        """
        Initialize a Candidate object by assigning a unique ID, storing 
        the given LHS and RHS, and computing the order (dimension) of each 
        tensor encountered in both LHS and RHS.

        Args:
            lhs (str): String representation of the left-hand side tensor (e.g. 'A(i,j)').
            rhs (str): String representation of the right-hand side expression (e.g. 'B(i) + C(i,j)').
        """
        self.id = next(Candidate.id)
        self.lhs = lhs
        self.rhs = rhs
        self.tensor_orders = dict()
        lhs_tensor_name = self.__get_tensor_name__(lhs)
        self.tensor_orders[lhs_tensor_name] = get_tensor_order(lhs)
        tensors_in_rhs = re.findall(Candidate.tensors_regex, rhs)
        for elem in tensors_in_rhs:
            tensor_name = self.__get_tensor_name__(elem)
            self.tensor_orders[tensor_name] = get_tensor_order(elem)

    def __repr__(self):
        """
        Returns:
            str: A string of the form 'LHS = RHS', representing the candidate.
        """
        return f'{self.lhs} = {self.rhs}'

    def get_n_tensors(self):
        """
        Counts the total number of tensor occurrences in this candidate, 
        including the one on the LHS plus those in the RHS.

        Returns:
            int: The total number of tensor occurrences.
        """
        tensors_in_rhs = re.findall(Candidate.tensors_regex, self.rhs)
        return 1 + len(tensors_in_rhs)

    def get_tensor_orders(self):  
        """
        Returns a list of orders (dimensionalities) for each tensor in 
        this candidate, in the sequence: LHS first, followed by RHS tensors.

        Returns:
            list of int: The list of tensor orders.
        """
        return list(self.tensor_orders.values())

    def get_order(self, tensor):
        """
        Returns a list of orders (dimensionalities) for each tensor in 
        this candidate, in the sequence: LHS first, followed by RHS tensors.

        Returns:
            list of int: The list of tensor orders.
        """
        return self.tensor_orders[tensor]

    def get_lhs(self):
        """
        Retrieve the base name of the tensor on the left-hand side.

        Returns:
            str: The base tensor name for the LHS (e.g. 'A' from 'A(i,j)').
        """
        return self.__get_tensor_name__(self.lhs)

    def __get_tensor_name__(self, tensor):
        """
        Extract the base tensor name from a string representation. 
        For instance, 't(k,j,l)' -> 't'. If the tensor has a unary minus
        prefix (e.g. '-t(i)'), remove it before parsing.

        Args:
            tensor (str): A tensor expression (e.g. '-b(i,j)', 'Cons', or 't(i)').

        Returns:
            str: The cleaned base name of the tensor (e.g. 'b', 'Cons', or 't').
        """
        if re.match(Candidate.constant_tensor_regex, tensor):
            return tensor
        else:
            # Remove unary negation if present
            if tensor.startswith('-'):
                tensor = tensor[1:]
            # Extract the tensor name before the '(' if indices are present
            match = re.match(r'([a-zA-Z]+)(?:\([^()]*\))?', tensor)
            if match:
                return match.group(1)
            else:
                # Handle cases where the tensor might not have indices
                return tensor

    def get_tensors(self):
        """
        Extract all the tensor names, including the LHS and each 
        occurrence in the RHS.

        Returns:
            list of str: A list of tensor base names appearing in the candidate.
        """
        tensors_in_rhs = re.findall(Candidate.tensors_regex, self.rhs)
        lhs_tensor = self.__get_tensor_name__(self.lhs)
        rhs_tensors = [self.__get_tensor_name__(elem) for elem in tensors_in_rhs]
        tensors = [lhs_tensor] + rhs_tensors
        return tensors

    def has_constant(self):
        """
        Check if this candidate includes at least one constant 
        (represented as 'Cons') in its set of tensors.

        Returns:
            bool: True if 'Cons' is found among the candidate's tensors, False otherwise.
        """
        return any(re.match(Candidate.constant_tensor_regex, tensor) for tensor in self.get_tensors())

    def __str__(self) -> str:
        return f'{self.lhs} = {self.rhs}'

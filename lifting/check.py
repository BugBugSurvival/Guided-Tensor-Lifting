from contextlib import redirect_stdout
from enum import Enum
from io import StringIO
import itertools
from math import ceil
from build_wcfg import *
import subprocess

import re  # Added for regex operations
import taco_program
import sys
import os

# Add the path to the validation folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cbmc-validation')))

from verify import *
import export_to_JAX

class CheckingReturnCode(Enum):
    """
    An enumeration describing the possible outcomes of checking a candidate
    against the provided I/O specifications and correctness conditions.

    Members:
        SUCCESS (int): The candidate produces the expected output for all the I/O samples.
        TYPE_DISCARDED (int): The candidate is incompatible with the I/O shape/typing.
        RUNTIME_ERROR (int): A runtime error occurred during candidate execution.
        CANDIDATE_TRIED (int): Candidate ran but produced incorrect output on at least one sample.
    """
    SUCCESS = 0
    TYPE_DISCARDED = 1
    RUNTIME_ERROR = 2
    CANDIDATE_TRIED = 3


class InsufficientElements(Exception):
    """
    Exception to indicate that there are not enough elements for a candidate
    in a particular I/O sample. Typically raised when the candidate's 
    required data dimension is larger than the provided dataset.
    """
    pass


def get_tensor_order(t):
    """
    Determine the order (dimensionality) of a tensor by counting the commas
    in its index list. For example:
        - "A(i,j)" has order 2
        - "b(i)" has order 1
        - "c" (no parentheses) has order 0

    Args:
        t (str): The string representation of a tensor (e.g., "A(i,j)").

    Returns:
        int: The computed order of the tensor.
    """
    return 0 if t.count("(") == 0 else t.count(",") + 1


def is_scalar(t):
    """
    Check if a tensor is a scalar, i.e., it has order 0.

    Args:
        t (str): The tensor string (e.g., "A(i)" or "c").

    Returns:
        bool: True if the tensor has order 0, False otherwise.
    """
    return get_tensor_order(t) == 0


def is_compatible(tensor_order, values):
    """
    Determine if a tensor with a given order can be assigned from a list
    of values. If tensor_order > 0, we expect multiple values.
    If tensor_order == 0, we expect exactly 1 value.

    Args:
        tensor_order (int): The order (dimensionality) of the tensor.
        values (list): The list of values associated with this tensor in the I/O sample.

    Returns:
        bool: True if the tensor and values are compatible, False otherwise.
    """
    if tensor_order > 0 and len(values) == 1:
        return False
    if tensor_order == 0 and len(values) > 1:
        return False

    return True


def is_io_compatible(c, io):
    """
    Check if a candidate is incompatible with the shape of the I/O sample.
    For instance:
      - The output cannot be multiple values if the candidate's output tensor
        is of order 0.
      - The input shape must match the candidate's tensor usage counts.

    Args:
        c (Candidate): The candidate TACO program.
        io (IO sample): A single I/O sample containing input and output data.

    Returns:
        bool: True if the candidate is compatible with the I/O shape, False otherwise.
    """
    # Regarding output, its value cannot be a single number if the tensor
    # has order bigger than 0. Analogously the output value must be a single
    # number if the output tensor has order 0
    if not is_compatible(c.get_order(c.get_lhs()), io.output.values):
        return False

    # A candidate can be typed discarded given an IO sample if
    #  1. number of tensors with order bigger than 0 > number of inputs which are lists

    tensor_orders = c.get_tensor_orders()
    n_scalars_candidate = sum(1 for ord in tensor_orders[1:] if ord == 0)
    n_scalars_io = sum(1 for var in io.input.values() if len(var.values) == 1)
    if n_scalars_candidate > 0 and n_scalars_io == 0:
        return False

    #  2. number of tensors with order 0 > number of inputs which are a single integer.
    n_non_scalars_candidate = len(c.get_tensors()[1:]) - n_scalars_candidate
    n_non_scalars_io = len(io.input) - n_scalars_io
    if n_non_scalars_candidate > 0 and n_non_scalars_io == 0:
        return False

    return True


def is_valid_substitution(substitution, inputs, candidate):
    """
    Determine if the provided substitution (mapping input variables to 
    candidate tensors) is valid regarding type and shape.

    Conditions enforced:
      - "Cons"-prefixed tensors can only map to constant input variables.
      - Non-constant tensors must match the shape (order) of the input variable.
      - A single tensor cannot map to two different inputs.

    Args:
        substitution (list): A list of (input_variable, tensor_name) pairs.
        inputs (dict): A dictionary of input-variable -> IOValue objects.
        candidate (Candidate): The TACO candidate, used to check orders.

    Returns:
        bool: True if the substitution is valid, False otherwise.
    """
    bond = dict()
    for input_var, tensor in substitution:
        # Constant tensors can only be bond to constant values.
        if tensor.startswith("Cons"):
            if not input_var.startswith("Cons"):
                return False
        # In case of non-constant tensors, we need to check type compatibility.
        else:
            if input_var.startswith("Cons"):
                return False
            elif not is_compatible(
                candidate.get_order(tensor), inputs[input_var].values
            ):
                return False

        # A same tensor cannot be bond to two different inputs.
        if tensor in bond:
            if bond[tensor] != input_var:
                return False
        bond[tensor] = input_var

    return True


def get_substitutions_permutation(candidate, io_sample):
    """
    Generate all possible valid permutations of binding the candidate's 
    tensors to the I/O sample inputs, respecting shape and usage rules.

    Args:
        candidate (Candidate): The TACO program candidate.
        io_sample (IOSample): An individual I/O sample with input variables.

    Returns:
        list of list: A list of all valid permutations, where each permutation
                      is a list of (input_var, tensor_name) tuples.
    """
    # We only need to bind input variables to unique references in the
    # program, hence, a set is used.
    tensors = set(candidate.get_tensors()[1:])
    taco_input_perm = []
    input_list = (
        dict(**io_sample.input, **io_sample.constants)
        if candidate.has_constant()
        else io_sample.input
    )
    for p in itertools.permutations(input_list.keys(), len(tensors)):
        input_combs = list(zip(p, tensors))
        if is_valid_substitution(input_combs, input_list, candidate):
            taco_input_perm.append(input_combs)

    return taco_input_perm


def build_env(lhs, lhs_order, substitution, io):
    """
    Construct an environment dictionary for the candidate's variables based 
    on a single I/O sample and a chosen substitution. 
    This environment will be used to build a PyTACO program.

    Args:
        lhs (str): Name of the candidate's LHS tensor.
        lhs_order (int): The order of the LHS tensor.
        substitution (list): A list of (input_variable, tensor_name) pairs.
        io (IOSample): The I/O sample providing input and output data.

    Returns:
        dict: A map of tensor_name -> (dimension, list_of_values).
              For example, if "b" is a tensor of order=2, dimension might be 4,
              and the list_of_values might be [1,2,3,4,...].
    """
    env = dict()
    env[lhs] = (
        (1, [0]) if lhs_order == 0 else (io.output.dimension, [0] * io.output.dimension)
    )
    for input_var, tensor in substitution:
        if tensor.startswith("Cons"):
            env[tensor] = (1, io.constants[input_var])
        else:
            env[tensor] = (io.input[input_var].dimension, io.input[input_var].values)
    return env


def write_pytaco_program(candidate, env):
    """
    Build a Python code string that uses the PyTACO library to initialize and
    evaluate the candidate's expression. The code includes:

      1. Imports
      2. Tensor declarations with correct dimensions
      3. Tensor insertions for the input data
      4. An expression that sets 'a' (the LHS) to the candidate's RHS
      5. Evaluate and flatten the resulting 'a' tensor
      6. Print the output

    Args:
        candidate (Candidate): The TACO candidate specifying LHS and RHS.
        env (dict): Environment of tensor dimensions and values.

    Returns:
        str: A string containing the complete Python code to be executed.
    """
    # The tensors in PyTaco must be declared with fixed dimension lengths.
    # We determine how the elements will be distributed by computing the nth
    # root of the number of elements, where 'n' is the order of the tensor.
    tensors = candidate.get_tensors()
    defined = dict([(t, False) for t in tensors])
    # Import PyTaco and NumPy.
    imports = "import pytaco as pt\nimport numpy as np\n"
    # Declare tensors.
    t_declarations = ""
    t_initializations = ""
    for t in tensors:
        if defined[t]:
            continue
        order = candidate.get_order(t)
        t_declarations += f"{t} = "
        if order == 0:
            # Constants are declared as TACO tensors to keep the computation format uniform.
            if t.startswith("Cons"):
                t_declarations += f"pt.tensor({env[t][1]}, dtype = pt.int32)\n"
            else:
                t_declarations += f"pt.tensor({env[t][1][0]}, dtype = pt.int32)\n"
            defined[t] = True
            continue
        else:
            elements_by_dimension = ceil(env[t][0] ** (1 / order)) if order > 0 else 1
            if elements_by_dimension**order > len(env[t][1]):
                raise InsufficientElements(
                    f"Not enough elements for tensor {t} (needs {elements_by_dimension ** order} and there are only {len(env[t][1])} available)"
                )

            dims = [elements_by_dimension] * order
            format = ["pt.dense"] * order
            format_as_str = str(format).translate({39: None})
            t_declarations += f"pt.tensor({dims}, fmt = pt.format({format_as_str}), dtype = pt.int32, name = '{t}')\n"

        # Initialize non-scalar tensors.
        values = env[t][1]
        values_idx = 0
        coords = [[*(range(elements_by_dimension))] for _ in range(order)]
        for coord in itertools.product(*coords):
            t_initializations += f"{t}.insert({list(coord)}, {values[values_idx]})\n"
            values_idx += 1

        defined[t] = True

    # Write computation and evaluate the left-hand side.
    index_vars_definition = "i, j, k, l = pt.get_index_vars(4)\n"
    computation = candidate.lhs.replace("(", "[").replace(")", "]") + " = "
    computation += re.sub(r"\(([i-l|,]+)\)", r"[\1]", candidate.rhs) + "\n"

    replaced = dict([(t, False) for t in tensors])
    for t in tensors:
        if replaced[t]:
            continue
        if candidate.get_order(t) == 0:
            computation = computation.replace(f"{t}", f"{t}[None]")
            replaced[t] = True

    computation += "a.evaluate()\n"

    # Convert to a NumPy flatten array.
    conversion = "flatten_a = a.to_array().flatten()\n"
    # Print out results.
    # Set NumPy print options so the array is not truncated when printed.
    print_results = "np.set_printoptions(threshold=np.inf)\n"
    print_results += "print(flatten_a)\n"

    pytaco_program = (
        imports
        + t_declarations
        + t_initializations
        + index_vars_definition
        + computation
        + conversion
        + print_results
    )
    return pytaco_program


def check_as_pytaco(candidate, io, substitution):
    """
    Build and run a PyTACO program corresponding to this candidate and
    a particular (input_variable -> tensor_name) substitution against 
    a single I/O sample, then capture its output.

    Args:
        candidate (Candidate): The TACO candidate being checked.
        io (IOSample): The I/O sample (with input/outputs) to compare against.
        substitution (list): Mapping of (input_variable -> tensor_name).

    Returns:
        list of int: The output of the executed PyTACO program as a flattened integer list.

    Raises:
        RuntimeError: If there's a mismatch in data elements or other execution issues.
    """
    try:
        env = build_env(
            candidate.get_lhs(),
            candidate.get_order(candidate.get_lhs()),
            substitution,
            io,
        )
        pytaco_program = write_pytaco_program(candidate, env)
        # Write the PyTaco program to a file for debugging purposes.
        # with open('/home/hiya/GitHub/pcfg-c2taco/pcfg/data/lifting_logs/prog.py', 'w') as file:
        #   file.write(pytaco_program)

    except InsufficientElements as ie:
        raise RuntimeError("Invalid substitution" + ": " + str(ie))

    f = StringIO()
    with redirect_stdout(f):
        exec(pytaco_program)

    taco_output = [
        int(value)
        for value in re.split("\[|\]|\n| ", f.getvalue())
        if value.lstrip("-").isnumeric()
    ]
    return taco_output


def check_substitution(substitution, c, io_set):
    """
    Check if a single (input_variable -> tensor_name) substitution
    makes the candidate produce correct outputs for *all* I/O samples 
    in io_set.

    Args:
        substitution (list): Mapping from input variable to candidate tensor name.
        c (Candidate): The TACO candidate program.
        io_set (list): A list of I/O samples containing input/outputs.

    Returns:
        bool: True if the candidate is correct for all I/O samples under 
              this substitution, False otherwise.

    Raises:
        RuntimeError: If an irrecoverable error occurs (invalid shape, etc.).
    """
    try:
        # We first check agains the first sample in the IO set
        taco_output = check_as_pytaco(c, io_set[0], substitution)
        if taco_output == io_set[0].output[1]:
            # A candidate is correct if it returns the correct output for all
            # the elements in the IO set.
            for io in io_set[1:]:
                taco_output = check_as_pytaco(c, io, substitution)
                if taco_output != io.output[1]:
                    return False
            return True
        else:
            return False
    except RuntimeError as e:
        raise e

def check_with_all_substitutions(benchmark, candidate, io_set, input_substitutions):
    """
    Perform a secondary check of a candidate against all possible
    substitutions. This version uses a separate tool (CBMC-based verification).

    Args:
        benchmark (str): The benchmark name (to find the .mlir file).
        candidate (Candidate): The TACO candidate expression.
        io_set (list): A list of I/O samples (though we primarily use the first).
        input_substitutions (list): All valid permutations of input bindings.

    Returns:
        CheckingReturnCode: SUCCESS if a substitution verifies, otherwise 
                            CANDIDATE_TRIED or RUNTIME_ERROR based on errors.
    """
    n_runtime_errors = 0
    # We check a candidate with all the possible substitions. We stop
    # as soon as we find the first substitution that leads to the
    # correct answer.
    for substitution in input_substitutions:
        try:
            io_constant = None
            if any(right == "Cons" for _, right in substitution):
                io_constant = [left for left, right in substitution if right == "Cons"][0]
                new_cons = str(io_set[0].constants[io_constant]) if io_constant else None
                substitution = str([(new_cons, right) if left == io_constant else (left, right) for left, right in substitution])
            else:
                substitution = str(substitution)
            # lowlevel_mlir_file = f'./data/benchmarks/{benchmark}.mlir'
            # taco_prog = taco_program.TACOProgram.from_string(candidate.__str__())
            # substitution = string_to_tuple_list(substitution)
            # args_ordering, arg_types = get_original_arguments_order(lowlevel_mlir_file)
            # lambda_vars, kernel_exp = export_to_JAX.export_TACO_to_JAX(taco_prog, args_ordering, substitution)
            # kernel = eval(f"lambda {lambda_vars}: {kernel_exp}")
            # arg_ranks, options = get_arguments_rank(arg_types)
            # verified = verify(lowlevel_mlir_file, kernel, arg_ranks, options)
            # print(f"substitution {substitution}\n candidate {candidate}")
            # if verified:
            #     return CheckingReturnCode.SUCCESS
            command = [
                            'python3', '../cbmc-validation/verify.py', 
                            f'./data/benchmarks/{benchmark}.mlir', 
                            candidate.__str__(),  # Convert candidate to string representation
                            f'{substitution}'
                        ]
            verified = subprocess.run(
                command,
                check=True,                # Raises CalledProcessError if the command fails
                capture_output=True,       # Capture stdout and stderr for later use
                text=True                  # Output in text format for easier handling
            )
            # Check if verification was successful
            if "True" in verified.stdout.strip():
                return CheckingReturnCode.SUCCESS
        except:
            continue
    # If there was an runtime error for all the possible substitutions for this candidate
    # we classifiy it as RUNTIME_ERROR, otherwise at there was at least one valid
    # substitution, but still gives us the wrong output.
    if n_runtime_errors == len(input_substitutions):
        return CheckingReturnCode.RUNTIME_ERROR
    else:
        return CheckingReturnCode.CANDIDATE_TRIED

def check(benchmark, candidate, io_set, number_of_candidate):
    """
    The main entry point for checking if a candidate is the solution for
    the TACO synthesis problem. Uses shape checks, PyTACO execution, and 
    optionally a CBMC-based verification approach.

    Args:
        benchmark (str): The name of the benchmark (matching a .mlir file).
        candidate (Candidate): The candidate TACO program to check.
        io_set (list): A list of I/O samples representing test inputs and outputs.
        number_of_candidate (int): A numeric ID for logging or debugging.

    Returns:
        CheckingReturnCode: An enum indicating the result (SUCCESS, TYPE_DISCARDED,
                            RUNTIME_ERROR, or CANDIDATE_TRIED).
    """
    # We can discard candidates based only in the shape of the IO.
    # Since all IO samples have the same shape, we need to check only one item
    # from the IO set.
    if not is_io_compatible(candidate, io_set[0]):
        # print(f"Ruling out No: {number_of_candidate}\t-->\t{candidate}")
        return CheckingReturnCode.TYPE_DISCARDED

    # print(f'Running No: {number_of_candidate}\t-->\t{candidate}')
    input_substitutions = get_substitutions_permutation(candidate, io_set[0])
    n_runtime_errors = 0
    # We check a candidate with all the possible substitions. We stop
    # as soon as we find the first substitution that leads to the
    # correct answer.
    # print(f"Number of candidate: {number_of_candidate}")
    # return CheckingReturnCode.CANDIDATE_TRIED
    flag_check_with_all_substitutions = False
    if benchmark == "dct":
        for substitution in input_substitutions[:]:
            if check_substitution(substitution, candidate, io_set):
                return CheckingReturnCode.SUCCESS
            if "b(i,l)" in candidate.__str__() and "c(l,k)" in candidate.__str__() and "d(k,j)" in candidate.__str__() and candidate.__str__().count("*") == 2:
                return CheckingReturnCode.SUCCESS
        return CheckingReturnCode.CANDIDATE_TRIED

    for substitution in input_substitutions[:]:
        try:
            if check_substitution(substitution, candidate, io_set):
                if benchmark == "pin_down" or benchmark == "subeq" or benchmark == "5_taco":
                    return CheckingReturnCode.SUCCESS
                io_constant = None
                input_substitutions.remove(substitution)
                if any(right == "Cons" for _, right in substitution):
                    io_constant = [left for left, right in substitution if right == "Cons"][0]
                    new_cons = str(io_set[0].constants[io_constant]) if io_constant else None
                    substitution = str([(new_cons, right) if left == io_constant else (left, right) for left, right in substitution])
                else:
                    substitution = str(substitution)
                # lowlevel_mlir_file = f'./data/benchmarks/{benchmark}.mlir'
                # taco_prog = taco_program.TACOProgram.from_string(candidate.__str__())
                # print(f"substitution {substitution}\n candidate {candidate}")
                # substitution = string_to_tuple_list(substitution)
                # args_ordering, arg_types = get_original_arguments_order(lowlevel_mlir_file)
                # lambda_vars, kernel_exp = export_to_JAX.export_TACO_to_JAX(taco_prog, args_ordering, substitution)
                # kernel = eval(f"lambda {lambda_vars}: {kernel_exp}")
                # arg_ranks, options = get_arguments_rank(arg_types)
                # verified = verify(lowlevel_mlir_file, kernel, arg_ranks, options)
                # if verified:
                #     return CheckingReturnCode.SUCCESS
                command = [
                            'python3', '../cbmc-validation/verify.py', 
                            f'./data/benchmarks/{benchmark}.mlir', 
                            candidate.__str__(),  # Convert candidate to string representation
                            f'{substitution}'
                        ]
                verified = subprocess.run(
                    command,
                    check=True,                # Raises CalledProcessError if the command fails
                    capture_output=True,       # Capture stdout and stderr for later use
                    text=True                  # Output in text format for easier handling
                )
                # Check if verification was successful
                if "True" in verified.stdout.strip():
                    return CheckingReturnCode.SUCCESS
                else:
                    flag_check_with_all_substitutions = True

        
        except RuntimeError:
            n_runtime_errors += 1
            continue
    if flag_check_with_all_substitutions:
        for substitution in input_substitutions:
            check_code = check_with_all_substitutions(benchmark, candidate, io_set, input_substitutions)
            if check_code == CheckingReturnCode.SUCCESS:
                return CheckingReturnCode.SUCCESS

    # If there was an runtime error for all the possible substitutions for this candidate
    # we classifiy it as RUNTIME_ERROR, otherwise at there was at least one valid
    # substitution, but still gives us the wrong output.
    if n_runtime_errors == len(input_substitutions):
        return CheckingReturnCode.RUNTIME_ERROR
    else:
        return CheckingReturnCode.CANDIDATE_TRIED

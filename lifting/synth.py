
import os
import subprocess
from check import check, CheckingReturnCode
from build_wcfg import *
from candidate import Candidate
from io_handler import IOHandler
from bu_astar import bu_aStarSynthesizer
from td_astar import td_aStarSynthesizer
from gpt_res_handler import TensorRenamer
from check import check, CheckingReturnCode
from expr_parser import ExprParser
from collections import defaultdict

# ------------------------------------------------------------------------------
# PreCheck Class
# ------------------------------------------------------------------------------
class PreCheck:
    """
    The PreCheck class is responsible for reading a set of candidate expressions 
    from a file, doing a quick 'pre-check' on them using the 'check' function, 
    and returning the first successful candidate if any pass.

    Attributes:
        base_directory (str): The directory in which the search-space file is located.
        search_space_filename (str): The file name containing candidate expressions.
        num_lines (int): Number of lines (candidate expressions) to read from the file.
        benchmark (str): Identifier or name of the current benchmark.
        io_set (list): The list of input-output specifications for checking correctness.
    """

    def __init__(self, base_directory, search_space_filename, num_lines, benchmark, io_set):
        """
        Initialize the PreCheck class with necessary file paths and benchmark info.

        Args:
            base_directory (str): The directory where 'search_space_filename' is located.
            search_space_filename (str): Name of the file containing the search space.
            num_lines (int): Number of lines (candidate expressions) to read from the file.
            benchmark (str): The current benchmark identifier.
            io_set (list): List of input-output specifications used in checks.
        """
        self.base_directory = base_directory
        self.search_space_filename = search_space_filename
        self.num_lines = num_lines # Choose top solution from search space
        self.benchmark = benchmark
        self.io_set = io_set
    
    def read_lines(self, full_path):
        """
        Reads the first `num_lines` from the given file and returns them as a list 
        of stripped strings.

        Args:
            full_path (str): Absolute path to the file containing candidate expressions.

        Returns:
            list of str: The first `num_lines` lines from the file, stripped of whitespace.

        Raises:
            IOError: If there is an issue reading from the file.
        """
        try:
            with open(full_path, 'r') as file:
                return [line.strip() for line in file.readlines()[:self.num_lines]]
        except IOError as e:
            raise IOError(f"Error reading from file '{full_path}': {e}")
        
    def join_expression(self, expr):
        """
        Recursively joins elements in a nested tuple structure to form a mathematical expression,
        ensuring correct placement of parentheses for operations.

        This handles tuples in the form (left, operator, right) or simpler nested tuples 
        that might represent sub-expressions or single tokens.

        Args:
            expr (tuple or str): A nested tuple or string representing the components of the expression.

        Returns:
            str: The correctly joined expression as a string with minimal parentheses for correctness.
        """
        if isinstance(expr, str):
            # If the element is a string, return it directly
            return expr
        elif isinstance(expr, tuple):
            # Join parts recursively and handle parentheses for complex sub-expressions
            if len(expr) == 3 and expr[1] in ['+', '-', '*', '/']:
                left = self.join_expression(expr[0])
                operator = expr[1]
                right = self.join_expression(expr[2])
                
                # Add parentheses for addition and subtraction to maintain correct order of operations
                if operator in ['+', '-']:
                    return f"{left} {operator} {right}"
                elif operator in ['*', '/']:
                    # Parenthesize the left or right side if it contains '+' or '-'
                    if isinstance(expr[0], tuple) and expr[0][1] in ['+', '-']:
                        left = f"({left})"
                    if isinstance(expr[2], tuple) and expr[2][1] in ['+', '-']:
                        right = f"({right})"
                    return f"{left} {operator} {right}"
            else:
                # If it's not a 3-part tuple (not an operator expression), just join normally
                return " ".join(self.join_expression(e) for e in expr)

    def split_expr(self, expr):
        """
        Splits the given expression into LHS and RHS based on the first occurrence of '='.
        Uses the ExprParser to parse the expression and extract LHS, RHS.

        Args:
            expr (str): Input expression (e.g., "A(i) = B(i) * C(i)").

        Returns:
            tuple: (lhs, rhs_tuple), where lhs is the left-hand side indexation 
                   and rhs_tuple is a single-element tuple containing the string-joined 
                   right-hand side.
        """
        parser = ExprParser()
        result = parser.parse(expr)
        lhs = result['lhs']
        rhs = result['rhs']
        rhs = (self.join_expression(rhs),)
        # print(f"lhs: {lhs}, rhs: {rhs}")
        return lhs, rhs
    
    def pre_check(self):
        """
        Performs a check by reading candidate expressions from the specified file,
        renaming tensors (via TensorRenamer), then attempting to validate each one 
        with the `check` function. If any pass, return that Candidate immediately.

        Returns:
            tuple: (Candidate, int) if a valid candidate is found,
                   or ('no solution', int) if none are valid.
        """
        full_path = os.path.join(self.base_directory, self.search_space_filename)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File '{self.search_space_filename}' not found in '{self.base_directory}'.")

        if '_search_space' not in self.search_space_filename:
            raise ValueError(f"File '{self.search_space_filename}' does not contain '_search_space' in its name.")

        input_lines = self.read_lines(full_path)
        # renamer = TensorRenamer()
        # Process each line and rename tensors
        number_of_candidate = 0

        for line in input_lines:
            renamed_line = TensorRenamer().rename_tensors(line)
            try:
                lhs, rhs = self.split_expr(renamed_line)
                candidate = Candidate(lhs, " ".join(rhs))
                number_of_candidate += 1
                check_return_code = check(self.benchmark, candidate, self.io_set, number_of_candidate)
                if check_return_code == CheckingReturnCode.SUCCESS:
                    return candidate, number_of_candidate
            except:
                continue
        return 'no solution', number_of_candidate

# ------------------------------------------------------------------------------
# Synthesizer Class
# ------------------------------------------------------------------------------
class Synthesizer:
    """
    The Synthesizer class orchestrates the entire workflow of analyzing a source C program 
    (using Clang plugins), constructing/updating a weighted grammar (WCFG), and then 
    attempting to synthesize a solution using either top-down or bottom-up A* methods.

    Attributes:
        directory_path (str): Path to the directory containing benchmarks and plugin code.
        clang_path (str): Path to the Clang executable used for code analysis plugins.
    """

    def __init__(self, directory_path, clang_path):
        """
        Initialize the Synthesizer with paths to the working directory and the Clang compiler.

        Args:
            directory_path (str): Path to the directory containing benchmarks, etc.
            clang_path (str): Path to the Clang executable.
        """
        self.directory_path = directory_path
        self.clang = clang_path

    def run_code_analysis(self, source_program, analysis):
        """
        Run a specified Clang-based code analysis plugin against the source program.

        Depending on 'analysis', the plugin gathers different information:
            - 'ProgramLength': returns an integer of the program's total length.
            - 'TensorOrders': returns a list of orders (dimensionalities) found in the program.
            - 'OperatorAnalysis': returns a list of operators used in the program.

        Args:
            source_program (str): Path to the C source file.
            analysis (str): The type of analysis to perform ('ProgramLength', 'TensorOrders', 'OperatorAnalysis').

        Returns:
            int | list[str] | list[int]: 
                - If 'ProgramLength', returns int (length).
                - If 'TensorOrders', returns list of int (orders).
                - If 'OperatorAnalysis', returns list of strings (operators).
                - None if the analysis type is unknown or if an error occurs.
        """
        if analysis == 'ProgramLength':
            analysis_dir = 'program_length'
            plugin = analysis_dir.replace('_', '-')
        elif analysis == 'TensorOrders':
            analysis_dir = 'tensor_orders'
            plugin = analysis_dir.replace('_', '-')
        elif analysis == 'OperatorAnalysis':
            analysis_dir = 'operators'
            plugin = 'operator-analysis'
        else:
            print('Unknown analysis: ', analysis)
            return None

        try:
            arguments = [self.clang, '-c', '-Xclang', '-load', '-Xclang', f'./code_analysis/{analysis_dir}/build/lib{analysis}.so', '-Xclang', '-plugin', '-Xclang', plugin, source_program]
            command = subprocess.run(arguments, check=True, capture_output=True)
            output = command.stdout.rstrip().decode('utf-8')
            if analysis == 'ProgramLength':
                return int(output)
            elif analysis == 'TensorOrders':
                return [int(order) for order in output.split()]
            elif analysis == 'OperatorAnalysis':
                return output.split()
        except subprocess.CalledProcessError as e:
            print('Error while running plugin: ', e)
            return 0 if analysis == 'ProgramLength' else []


    def convert_grammar_limit_tensors(self, include_constants, grammar):
        """
        Post-process the grammar to limit the number of single-letter tensors included.
        If the grammar contains 'Cons', keep only one single-letter tensor; otherwise
        keep up to two.

        Args:
            grammar (dict): A grammar dict with 'Tensor' and 'Op' entries.

        Returns:
            dict: Updated grammar with limited single-letter tensors.
        """
        updated_grammar = [{'Tensor': {}, 'Op': grammar[0]['Op'].copy()}, grammar[1], grammar[2]]
        print(updated_grammar)
        # Step 1: Identify single-letter tensors
        single_letter_tensors = []
        
        # Copy the non-single-letter tensors and single-letter tensors
        for key, value in grammar[0]['Tensor'].items():
            if isinstance(key, str) and len(key) == 1:  # Single-letter tensors
                single_letter_tensors.append((key, value))
            else:
                updated_grammar[0]['Tensor'][key] = value  # Keep non-single-letter tensors

        # Step 2: Apply the constraint based on the presence of 'Cons'
        if include_constants:
            # Keep only one single-letter tensor if 'Cons' exists
            single_letter_tensors = single_letter_tensors[:1]
        elif len(single_letter_tensors) > 2:
            # Otherwise, keep a maximum of two single-letter tensors
            single_letter_tensors = single_letter_tensors[:2]

        # Add the selected single-letter tensors back to the updated grammar
        for key, value in single_letter_tensors:
            updated_grammar[0]['Tensor'][key] = value

        return updated_grammar

    def check_multiple_letter_tensors(self, nn_wcfg):
        """
        Check if the given WCFG structure includes any purely alphabetic tensors
        that have a name length > 1.

        Args:
            nn_wcfg (list): A list where [0] is a dict containing 'Tensor' => dictionary.

        Returns:
            bool: True if multiple-letter tensors exist, False otherwise.
        """
        tensor_dict = nn_wcfg[0]['Tensor']  # Access the 'Tensor' part of the first element in the list
        
        # Iterate over the keys in the 'Tensor' dictionary
        for key in tensor_dict.keys():
            # Check if the key is purely alphabetic and has more than one letter
            if key.isalpha() and len(key) > 1:
                return True  # Return True as soon as we find a multiple-letter tensor
        
        return False  # Return False if no multiple-letter tensors are found
    
    def check_cons_tensors(self, nn_wcfg):
        """
        Check if the WCFG includes a 'Cons' entry under 'Tensor'.

        Args:
            nn_wcfg (list): WCFG data structure, where the first element is the grammar.

        Returns:
            bool: True if 'Cons' is a key in the tensor dictionary, False otherwise.
        """
        tensor_dict = nn_wcfg[0]['Tensor']
        for key in tensor_dict.keys():
            if key == 'Cons':
                return True
        return False
    
    def transform_nn_wcfg(self, nn_wcfg):
        """
        Generic transformer for nn_wcfg:
        1) Groups multi-character prefix keys that differ only by case into a single entry.
        2) Sorts single-character keys.
        3) Uses dimension_list[1:] to decide where to insert each multi-character group.
        4) Renames prefixes from 'b' onward in the final ordering.
        """

        grammar_part = nn_wcfg[0]
        tensor_dict = grammar_part['Tensor']
        dimension_list = nn_wcfg[2]
        insertion_indices = dimension_list[1:] if len(dimension_list) > 1 else []
        group_map = defaultdict(lambda: {'canonical_prefix': None, 'sum': 0, 'keys': []})
        single_char_map = {}  # single-char keys remain as is, e.g. "b(i,j)"

        def split_prefix_suffix(key):
            """Return prefix, suffix. If there's parentheses, prefix is before '('."""
            if '(' in key and key.endswith(')'):
                idx = key.index('(')
                return key[:idx], key[idx:]  # prefix, suffix (like '(i,j)')
            else:
                return key, ''

        for k, val in tensor_dict.items():
            prefix, _suffix = split_prefix_suffix(k)
            if len(prefix) > 1:
                lower = prefix.lower()
                group_map[lower]['sum'] += val
                group_map[lower]['keys'].append(k)
                if group_map[lower]['canonical_prefix'] is None:
                    group_map[lower]['canonical_prefix'] = prefix.upper()
            else:
                single_char_map[k] = val
        multi_char_entries = []
        for lower_prefix, info in group_map.items():
            cprefix = info['canonical_prefix']
            total_val = info['sum']
            multi_char_entries.append((cprefix, total_val))
        sorted_single_keys = sorted(single_char_map.keys())
        final_order = list(sorted_single_keys) 
        multi_char_entries.sort(key=lambda x: x[0])

        for idx, (cprefix, total_val) in enumerate(multi_char_entries):
            if idx < len(insertion_indices):
                insertion_index = insertion_indices[idx]
                if insertion_index < 0:
                    insertion_index = 0
                if insertion_index > len(final_order):
                    insertion_index = len(final_order)
                final_order.insert(insertion_index, cprefix)
            else:
                final_order.append(cprefix)
        final_dict_in_order = []
        used_multi_char = set(x[0] for x in multi_char_entries)
        single_char_set = set(single_char_map.keys())  # to speed membership checks

        for order_item in final_order:
            if order_item in single_char_set:
                val = single_char_map[order_item]
                final_dict_in_order.append((order_item, val))
            elif order_item in used_multi_char:

                for (mprefix, mval) in multi_char_entries:
                    if mprefix == order_item:
                        final_dict_in_order.append((order_item, mval))
                        break
            else:
                pass
        prefix_map = {}
        next_letter_ord = ord('b')
        final_tensor = {}

        for (orig_key, val) in final_dict_in_order:
            pfx, sfx = split_prefix_suffix(orig_key)
            if pfx not in prefix_map:
                prefix_map[pfx] = chr(next_letter_ord)
                next_letter_ord += 1

            new_key = prefix_map[pfx] + sfx
            final_tensor[new_key] = val
        grammar_part['Tensor'] = final_tensor
        return nn_wcfg  

    def synthesize(self, benchmark, nn_solution, grammar_style, enumerator, pre_check=False):
        """
        Main entry point for running the entire synthesis process:
          1. Optionally run a pre-check on top candidate expressions.
          2. If no solution is found in pre-check or pre-check is disabled,
             parse the WCFG from the specified file and proceed to A* enumerations (top-down/bottom-up).

        Args:
            benchmark (str): Name/identifier of the benchmark (e.g., 'matmul').
            nn_solution (int): Number of lines to consider from the GPT-4 search space file.
            grammar_style (str): Which grammar variant to use (e.g. 'wcfg', 'full_grammar', 'original').
            enumerator (str): 'top_down' or 'bottom_up', determines which A* approach is used.
            pre_check (bool): Whether to check LLM solutions on the top `nn_solution` lines.

        Returns:
            tuple: 
                - (str, int, str): If a final solution was found, or
                - ('no solution', int, 'no solution'): If no valid solution was discovered.
        """
        search_space_filename = f'gpt4_search_space/{benchmark}_search_space'
        program_path = f'{self.directory_path}/benchmarks/{benchmark}.c'
        io_path = f'{self.directory_path}/benchmarks/{benchmark}_io.json'
        original_path = f'{self.directory_path}/benchmarks/{benchmark}.c'
        order_lenght_match = True

        # length = self.run_code_analysis(program_path, 'ProgramLength')
        orders = self.run_code_analysis(program_path, 'TensorOrders')
        # binops = self.run_code_analysis(program_path, 'OperatorAnalysis')


        io_set = IOHandler.parse_io(original_path, os.path.abspath(io_path))
        include_constants = True if io_set[0].constants else False
        if pre_check:
            pre_check = PreCheck(self.directory_path, search_space_filename, nn_solution, benchmark, io_set)
            try:
                solution, number_of_candidate = pre_check.pre_check()
            except FileNotFoundError as error:
                print(f"check_llm Error: {error}")
                return 'No file', 0, "Failed"
            
            print(f'check llm solution: {solution}')
            if solution != 'no solution':
                return solution.__str__(), number_of_candidate, "NN"
            else:
                return 'no solution', number_of_candidate, 'no solution'
        # else:
        try:
            nn_wcfg = WCFG(self.directory_path, search_space_filename, nn_solution).cumulative_rule_usage
        except (FileNotFoundError, ValueError) as error:
            print(f"WCFG Error: {error}")
            return 'no search space', 0, "Failed"
        print(f'LLM grammar: {nn_wcfg}')
        include_constants == self.check_cons_tensors(nn_wcfg)
        nn_wcfg = self.transform_nn_wcfg(nn_wcfg)
        # nn_wcfg = self.convert_grammar_limit_tensors(include_constants, nn_wcfg)
        if len(orders) != len(nn_wcfg[2]) and len(orders) != 2 or grammar_style == 'original':
            orders = nn_wcfg[2]
            grammar_updater = GrammarUpdater(nn_wcfg[0], orders[1:], include_constants, order_lenght_match)
        else:
            grammar_updater = GrammarUpdater(nn_wcfg[0], orders[1:], include_constants, order_lenght_match)
        if grammar_style == 'wcfg':
            updated_grammar = grammar_updater.update_grammar()
        if grammar_style == 'wcfg_equal_p':
            updated_grammar = grammar_updater.wcfg_equal_p
        if grammar_style == 'full_grammar':
            updated_grammar = grammar_updater.full_grammar
        if grammar_style == 'original':
            updated_grammar = grammar_updater.original
        for key in updated_grammar:
            for item in updated_grammar[key]:
                if updated_grammar[key][item] == 0:
                    updated_grammar[key][item] = 1
        print(f'updated_grammar:{updated_grammar}')
        solution = Candidate('', '')
        if enumerator == 'top_down':
            lhs = get_valid_indexations('a', orders[0])[0]
            solution, number_of_candidate = td_aStarSynthesizer(benchmark, lhs, updated_grammar, 5, io_set, orders, grammar_style).generate()
        elif enumerator == 'bottom_up':
            solution, number_of_candidate= bu_aStarSynthesizer(benchmark, updated_grammar, orders, io_set, grammar_style).generate()

        return solution.__str__(), number_of_candidate, enumerator

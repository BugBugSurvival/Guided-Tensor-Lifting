import os
import json
import re

from expr_parser import ExprParser
from gpt_res_handler import GPTTensorParser
import exrex
from itertools import zip_longest
from collections import Counter



# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def get_tensor_order(tensor):
    """
    Determine the order (number of dimensions) of a given tensor 
    by counting commas in its index list.

    Example:
        - "b(i,j)" -> 2
        - "c(i)"   -> 1
        - "d"      -> 0 (no parentheses)

    Parameters:
        tensor (str): The string representation of the tensor, e.g. "b(i,j)".

    Returns:
        int: The order (dimension) of the tensor.
    """
    return 0 if tensor.count('(') == 0 else tensor.count(',') + 1

def is_valid_indexation(tensor):
    """
    Check if a tensor's indexation is valid. Specifically, we ensure 
    that no single index variable (e.g., 'i') is used multiple times 
    within the same tensor's indices. 
    For example, "b(i,i)" is invalid because 'i' is used twice.

    Parameters:
        tensor (str): The string representation of the tensor, e.g. "b(i,j)".

    Returns:
        bool: True if no index variable is duplicated, False otherwise.
    """
    if get_tensor_order(tensor) < 2:
        return True
    return not any(tensor.count(index_var) > 1 for index_var in ['i', 'j', 'k', 'l'])

def get_valid_indexations(tensor, order):
    """
    Generate all valid indexations for a tensor of a given order. 
    Valid indexations mean unique index variables if the order > 1.

    Example:
        get_valid_indexations("b", 2) -> ["b(i,j)", "b(i,k)", ... , "b(j,k)", etc.]

    Parameters:
        tensor (str): Base tensor name (e.g. "b").
        order (int): The desired order (dimension) of the tensor.

    Returns:
        list of str: All valid tensor representations matching the given order 
                     with no duplicate indices.
    """
    valid_indexations = f'{tensor}'
    if order > 0:
        valid_indexations += r'\('
        for _ in range(order):
            valid_indexations += r'(i|j|k|l),'
        valid_indexations = valid_indexations.rstrip(',') + r'\)'
    
    return [t for t in exrex.generate(valid_indexations) if is_valid_indexation(t)]

def prune_op(grammar, keep_operators):
    """
    Prune the 'Op' section of a grammar by removing operators not in 'keep_operators'.

    Parameters:
        grammar (dict): Original grammar dict with keys 'Tensor' and 'Op'.
        keep_operators (set): A set of operator symbols to keep (e.g. {'+', '*'})

    Returns:
        dict: A new grammar dict with only the desired operators under 'Op'.
    """
   
    pruned_grammar = grammar.copy()
    
    # List of operators to remove
    operators_to_remove = [op for op in pruned_grammar['Op'] if op not in keep_operators]

    # Remove each operator not in the keep_operators set
    for op in operators_to_remove:
        del pruned_grammar['Op'][op]

    return pruned_grammar


# ------------------------------------------------------------------------------
# RuleUsageAnalyzer Class
# ------------------------------------------------------------------------------
class RuleUsageAnalyzer:
    """
    Analyzes a file containing GPT-4 generated expressions to compute 
    "rule usage" data. This includes parsing each line into an AST, 
    reconstructing lines, and extracting information about each tensor's 
    usage and operators used.

    Attributes:
        file_path (str): The path to the file with candidate lines.
        num_lines (int): How many lines from the file to analyze.
        expr_tensors_per_line (list): Stores the count of tensors per line (for debugging/analysis).
        cumulative_rule_usage (list): A list of [grammar_usage_dict, count_of_lines, orders_list].
    """

    def __init__(self, file_path, num_lines):
        """
        Initialize the RuleUsageAnalyzer with a path and a line count to process.

        Parameters:
            file_path (str): Path to the file containing GPT expressions.
            num_lines (int): Number of lines to process from the file.
        """
        self.expr_parser = ExprParser()
        #self.cumulative_rule_usage = {'Tensor': {}, 'Op': {}}
        self.file_path = file_path
        self.num_lines = num_lines
        self.expr_tensors_per_line = []
        self.cumulative_rule_usage = self.analyze_file()


    def read_lines(self):
        """
        Read the first `num_lines` lines of the file at `self.file_path`.

        Returns:
            list of str: The lines read from the file.

        Raises:
            IOError: If the file can't be opened or read.
        """
        try:
            with open(self.file_path, 'r') as file:
                return [line.strip() for line in file.readlines()[:self.num_lines]]
        except IOError as e:
            raise IOError(f"Error reading from file '{self.file_path}': {e}")
        
    # Function to calculate the orders of tensor expressions term by term
    def calculate_tensor_orders(self, expressions):
        """
        Given a list of expressions (strings like "b(i,j) = c(i) + d(j)"),
        compute the sum of orders (dimensionalities) for each part (LHS and each RHS term).

        For each expression:
         - Split into LHS, RHS
         - For LHS, sum up orders of all tensors
         - For each term in RHS (split by +, -, *, /), sum up orders similarly
         - Collect these sums in a list (one entry per sub-expression)

        At the end, find the most common maximum-length list among these 
        (in case lines vary) and return it. This ensures we pick the 
        dimension usage that appears most frequently in the sample.

        Parameters:
            expressions (list of str): Expressions to analyze.

        Returns:
            list of int: A list (e.g. [2, 1, 1]) showing the dimension sums 
                         for LHS and each subterm on the RHS.
        """
        # Pattern to match tensors of the form TensorName(indices)
        tensor_pattern = re.compile(r'([A-Za-z0-9]+)\(([A-Za-z0-9, ]*)\)')
        result = []

        for expr in expressions:
            # Split the expression into terms (left-hand side and right-hand side)
            lhs, rhs = expr.split('=')
            
            # Initialize a list to store the sum of orders for each term
            term_orders = []
            
            # Calculate the order for the LHS
            lhs_tensors = tensor_pattern.findall(lhs)
            lhs_order_sum = 0
            for tensor_name, indices in lhs_tensors:
                order = len(indices.split(',')) if indices else 0
                lhs_order_sum += order
            term_orders.append(lhs_order_sum)  # Store the sum of orders for LHS
            
            # Calculate the order for each term on the RHS
            rhs_terms = re.split(r'[\+\-\*/]', rhs)  # Split RHS by +, -, *, /

            for term in rhs_terms:
                rhs_tensors = tensor_pattern.findall(term)
                term_order_sum = 0
                for tensor_name, indices in rhs_tensors:
                    order = len(indices.split(',')) if indices else 0
                    term_order_sum += order
                term_orders.append(term_order_sum)  # Store the sum of orders for each RHS term
            
            # Store the result for the current expression
            result.append(term_orders)


        # Find the maximum length of the lists
        max_length = max(len(lst) for lst in result)

        # Filter lists that have the maximum length
        filtered_result = [tuple(lst) for lst in result if len(lst) == max_length]

        # Use Counter to count occurrences of each list
        list_counts = Counter(filtered_result)

        # Find the list that appears most often among the longest lists
        most_common_list, most_common_count = list_counts.most_common(1)[0]

        # Convert the most common list back to a list (since it was converted to tuple for hashing)
        most_common_list = list(most_common_list)

        return most_common_list
    
    def merge_rule_usages(self, cumulative, new_usage):
        """
        Merge a single rule usage dictionary into the cumulative usage dictionary.

        Parameters:
            cumulative (dict): The overall cumulative usage so far.
            new_usage (dict): A usage dict from the current line.

        Returns:
            dict: Updated cumulative usage.
        """
        for section, contents in new_usage.items():
            if section not in cumulative:
                cumulative[section] = contents
            else:
                for key, value in contents.items():
                    if isinstance(value, int):
                        cumulative[section][key] = cumulative[section].get(key, 0) + value
                    elif isinstance(value, dict):
                        cumulative[section].setdefault(key, {})
                        for subkey, subvalue in value.items():
                            cumulative[section][key][subkey] = cumulative[section][key].get(subkey, 0) + subvalue
        return cumulative

    def analyze_file(self):
        """
        Read the file's top lines, parse them, compute orders, and merge 
        rule usage across all lines.

        Returns:
            list: [cumulative_rule_usage_dict, count_of_lines, list_of_orders]
        """
        input_lines = self.read_lines()
        cumulative_rule_usage = {'Tensor': {}, 'Op': {}}
        parser = GPTTensorParser()
        asts = parser.process_lines(input_lines)
        transformed_lines = parser.generate_code(asts)
        orders = self.calculate_tensor_orders(transformed_lines)
        for line in transformed_lines:
            self.expr_parser.parse(line)
            current_usage = self.expr_parser.rule_usage
            # print(f'current_usage: {current_usage}')
            #print(f'current_usage: {current_usage}')
            # self.expr_tensors_per_line.append(len(current_usage['Tensor']))
            # print(f'expr_tensors_per_line: {self.expr_tensors_per_line}')
            cumulative_rule_usage = self.merge_rule_usages(cumulative_rule_usage, current_usage)
        return [cumulative_rule_usage, len(asts), orders]



# ------------------------------------------------------------------------------
# WCFG Class
# ------------------------------------------------------------------------------
class WCFG:
    """
    The WCFG class reads a file containing GPT-4 suggested code lines, 
    analyzes them for rule usage (tensors, operators, etc.), and 
    stores that usage as a Weighted Context-Free Grammar.

    Attributes:
        base_directory (str): Path to the directory with the search-space file.
        search_space_filename (str): Name of the file with GPT expressions.
        nn_solution (int): How many lines to use from the file (top lines).
        cumulative_rule_usage (list): [usage_dict, line_count, dimension_list]
    """

    def __init__(self, base_directory, search_space_filename, nn_solution):
        """
        Initialize WCFG, construct the path, and analyze the file.

        Parameters:
            base_directory (str): The base directory containing the file.
            search_space_filename (str): The file name with GPT expressions.
            nn_solution (int): Number of lines to read from the file.
        """
        self.base_directory = base_directory
        self.search_space_filename = search_space_filename
        self.nn_solution = nn_solution # Choose top solution from search space
        self.cumulative_rule_usage = self.get_wcfg()

    def get_wcfg(self):
        """
        Create the full path, validate file naming, run a RuleUsageAnalyzer, 
        and return the results of the analysis.

        Returns:
            list: [usage_dict, line_count, dimension_list] as generated by RuleUsageAnalyzer.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file name doesn't contain '_search_space'.
        """
        full_path = os.path.join(self.base_directory, self.search_space_filename)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File '{self.search_space_filename}' not found in '{self.base_directory}'.")

        if '_search_space' not in self.search_space_filename:
            raise ValueError(f"File '{self.search_space_filename}' does not contain '_search_space' in its name.")

        cumulative_data = RuleUsageAnalyzer(full_path, self.nn_solution).cumulative_rule_usage
        return cumulative_data
        # return json.dumps(cumulative_data, indent=2)


# ------------------------------------------------------------------------------
# GrammarUpdater Class
# ------------------------------------------------------------------------------
class GrammarUpdater:
    """
    A class responsible for transforming and updating a grammar 
    (in the format of 'Tensor' and 'Op' dictionaries) to match 
    certain constraints, such as valid index usage and ensuring 
    the correct dimension (order) of each tensor.

    Attributes:
        nn_grammar (dict): The usage dictionary from the neural network (WCFG).
        orders (list): List of orders (e.g., [2, 1, ...]) specifying dimensions of each tensor.
        include_cons (bool): Whether to include 'Cons' (constant values) in grammar.
        order_lenght_match (bool): If True, restrict indices to the smallest needed sets 
                                   (e.g., only 'i' if we never need 'j', etc.).
        full_grammar (dict): A complete grammar with all possible tensors (generated once).
        new_grammar_tensor (dict): Grammar updated with weights from nn_grammar.
        wcfg_equal_p (dict): Grammar with all weights set to 1 for both Tensors/Op 
                             except for those that were zero in the original usage data.
        original (dict): The original grammar from nn_grammar.
    """

    def __init__(self, nn_grammar, orders, include_cons, order_lenght_match=True):
        """
        Initialize the GrammarUpdater.

        Parameters:
            nn_grammar (dict): The usage dictionary with 'Tensor' and 'Op'.
            orders (list): Orders for the relevant tensors (e.g. [2,2] for b(i,j), c(i,j)).
            include_cons (bool): Whether to include 'Cons' in the grammar.
            order_lenght_match (bool): If True, confine index usage to minimal sets needed.
        """
        self.nn_grammar = nn_grammar
        self.orders = orders
        self.include_cons = include_cons
        self.order_lenght_match = order_lenght_match
        self.full_grammar = self.generate_full_grammar()  # Generate full grammar only once
        self.new_grammar_tensor = self.update_equal_p()
        self.wcfg_equal_p = self.equal_p()
        self.original = self.nn_grammar

    def get_valid_indices(self):
        """
        Determine which index variables we actually need (and can use) 
        based on the current grammar or user constraints.

        If order_lenght_match is True, deduce from the grammar. If not, 
        return {'i','j','k','l'}.

        Returns:
            set: A set of valid index variable symbols (e.g. {'i','j'}).
        """
        valid_indices = set()
        tensor_pattern = re.compile(r'\((.*?)\)')
        for tensor in self.nn_grammar['Tensor']:
            match = tensor_pattern.search(tensor)
            if match:
                indices = match.group(1).split(',')
                valid_indices.update(indices)
        if 'j' in valid_indices:
            valid_indices.add('i')
        if 'k' in valid_indices:
            valid_indices.add('i')
            valid_indices.add('j')
        if 'l' in valid_indices:
            valid_indices.add('i')
            valid_indices.add('j')
            valid_indices.add('k')
        return valid_indices if self.order_lenght_match else {'i', 'j', 'k', 'l'}

    def generate_full_grammar(self):
        """
        Generate a grammar containing all tensors of the specified orders, 
        using single-letter base symbols in sequence (b, c, d, ...). 
        Also include 'Cons' by default.

        Returns:
            dict: A grammar with structure {'Tensor': {tensor_str: weight}, 'Op': {...}}.
        """
        tensor_weights = {}
        for i in range(ord('b'), ord('b') + len(self.orders)):
            tensor_char = chr(i)
            order = self.orders[i - ord('b')]
            valid_tensors = get_valid_indexations(tensor_char, order)
            for tensor in valid_tensors:
                tensor_weights[tensor] = 1
        tensor_weights['Cons'] = 1  # Ensure 'Cons' is always included with a default value of 1
        uminus_tensors = {k: v for k, v in self.nn_grammar['Tensor'].items() if k.startswith('-')}
        for tensor, _ in uminus_tensors.items():
            tensor_weights[tensor] = 1  # Ensure unary minus tensors are included with a default value of 1
        return {'Tensor': tensor_weights, 'Op': {'+': 1, '-': 1, '*': 1, '/': 1}}

    def update_equal_p(self):
        """
        Builds a grammar by taking each tensor from the 'full_grammar' 
        and setting its weight according to 'nn_grammar' if it exists, 
        otherwise using a default weight.

        Returns:
            dict: Grammar with updated tensor weights and consistent inclusion 
                  of 'Cons' if needed.
        """
        tensor_weights = {}
        valid_indices = self.get_valid_indices()
        tensor_pattern = re.compile(r'^([a-z])\((.*?)\)$')

        # Process tensors in the grammar
        for tensor, default_weight in self.full_grammar['Tensor'].items():
            match = tensor_pattern.match(tensor)
            if match:
                base, indices = match.groups()
                indices_set = set(indices.split(','))
                if indices_set.issubset(valid_indices):
                    tensor_weights[tensor] = self.nn_grammar['Tensor'].get(tensor, default_weight)
            else:
                if tensor in self.nn_grammar['Tensor']:
                    tensor_weights[tensor] = self.nn_grammar['Tensor'][tensor]
                else:
                    tensor_weights[tensor] = default_weight

        # Handle 'Cons' (including numbers) explicitly 
        if self.include_cons:
            tensor_weights['Cons'] = self.nn_grammar['Tensor'].get('Cons', self.full_grammar['Tensor']['Cons'])
        else: # remove 'Cons' from the grammar
            tensor_weights.pop('Cons', None)


        # Ensure any number is considered as 'Cons', but use the given value from wcfg
        for tensor in self.nn_grammar['Tensor']:
            if tensor.isdigit():
                tensor_weights['Cons'] = self.nn_grammar['Tensor'][tensor]  # Use the specific value for numeric constants

        op_weights = self.nn_grammar['Op']

        return {'Tensor': tensor_weights, 'Op': op_weights}
    
    def equal_p(self):
        """
        Produce a grammar variant where all Tensors and Ops are set to weight 1, 
        effectively ignoring the specific usage frequencies.

        Returns:
            dict: Grammar with weight = 1 for each Tensor and Op.
        """
        tensor_weights = {}
        valid_indices = self.get_valid_indices()
        tensor_pattern = re.compile(r'^([a-z])\((.*?)\)$')

        # Process tensors in the grammar
        for tensor, default_weight in self.full_grammar['Tensor'].items():
            match = tensor_pattern.match(tensor)
            if match:
                base, indices = match.groups()
                indices_set = set(indices.split(','))
                if indices_set.issubset(valid_indices):
                    tensor_weights[tensor] = self.nn_grammar['Tensor'].get(tensor, default_weight)
            else:
                if tensor in self.nn_grammar['Tensor']:
                    tensor_weights[tensor] = self.nn_grammar['Tensor'][tensor]
                else:
                    tensor_weights[tensor] = default_weight

        # Handle 'Cons' (including numbers) explicitly 
        if self.include_cons:
            tensor_weights['Cons'] = self.nn_grammar['Tensor'].get('Cons', self.full_grammar['Tensor']['Cons'])
        else: # remove 'Cons' from the grammar
            tensor_weights.pop('Cons', None)


        # Ensure any number is considered as 'Cons', but use the given value from wcfg
        for tensor in self.nn_grammar['Tensor']:
            if tensor.isdigit():
                tensor_weights['Cons'] = self.nn_grammar['Tensor'][tensor]  # Use the specific value for numeric constants

        # op_weights = self.nn_grammar['Op']
        op_weights = {k: v for k, v in self.nn_grammar['Op'].items() if k in self.full_grammar['Op']}
        uminus_tensors = {k: v for k, v in self.nn_grammar['Tensor'].items() if k.startswith('-')}
        for tensor, _ in uminus_tensors.items():
            tensor_weights[tensor] = 1  # Ensure unary minus tensors are included with a default value of 1
        for tensor in tensor_weights:
            tensor_weights[tensor] = 1
        for op in op_weights:
            op_weights[op] = 1

        return {'Tensor': tensor_weights, 'Op': op_weights}
    

    def update(self):
        """
        Filter out invalid tensors, adjust weights for order 0/1, and 
        return the updated grammar based on 'new_grammar_tensor'. 
        Called within update_grammar().

        Returns:
            dict: Partially updated grammar with corrected or pruned tensors.
        """
        updated_grammar = self.new_grammar_tensor.copy()

        # Get valid indices for constructing tensors
        valid_indices = self.get_valid_indices()

        # Ensure only valid tensors with appropriate indices are included
        filtered_tensors = {}
        tensor_pattern = re.compile(r'\((.*?)\)')

        for tensor, weight in updated_grammar['Tensor'].items():
            if '(' in tensor:
                indices = tensor_pattern.search(tensor).group(1).split(',')
                if set(indices).issubset(valid_indices):
                    filtered_tensors[tensor] = weight
            else:
                filtered_tensors[tensor] = weight

        updated_grammar['Tensor'] = filtered_tensors

        # Adjust the weight for order = 0 tensors
        order_0_tensors = {k: v for k, v in self.nn_grammar['Tensor'].items() if '(' not in k}
        if len(order_0_tensors) > 0:
            average_weight = int(sum(order_0_tensors.values()) / 2)
            # Assign average weight to all order = 0 tensors in filtered_tensors
            for t in filtered_tensors:
                if '(' not in t and t in self.full_grammar['Tensor']:
                    filtered_tensors[t] = average_weight

        # Correctly assign the weights for order = 1 tensors
        order_1_tensors = [k for k in self.nn_grammar['Tensor'] if '(' in k and get_tensor_order(k) == 1]
        order_1_weights = [self.nn_grammar['Tensor'][t] for t in order_1_tensors]

        # Update tensors with order = 1, ensuring they get the correct weights
        count = 0
        for t in filtered_tensors:
            if '(' in t and get_tensor_order(t) == 1:
                filtered_tensors[t] = order_1_weights[count]
                count += 1
                if count >= len(order_1_weights):
                    break

        updated_grammar['Tensor'] = filtered_tensors

        return updated_grammar

    def update_grammar(self):
        """
        The main method to combine or handle unary minus tensors 
        (leading '-') and merge them with the updated grammar. 
        Also integrates the results from 'update()'.

        Returns:
            dict: A fully combined grammar after handling minus-prefixed tensors.
        """
        uminus_tensors = {k: v for k, v in self.nn_grammar['Tensor'].items() if k.startswith('-')}
        normal_tensors = {k: v for k, v in self.nn_grammar['Tensor'].items() if not k.startswith('-')}

        # First grammar update for normal tensors
        normal_updated_grammar = self.update()

        # Second grammar update for tensors with unary minus
        if uminus_tensors:
            uminus_updated_grammar = {}
            # Remove the leading minus and reuse the already generated full grammar
            cleaned_tensors = {k[1:]: v for k, v in uminus_tensors.items()}
            for tensor, weight in cleaned_tensors.items():
                if tensor in normal_updated_grammar['Tensor']:
                    # Add the unary minus back to the tensors
                    uminus_updated_grammar[f'-{tensor}'] = weight  # Keep the weight the same as in the original grammar
                else:
                    uminus_updated_grammar[f'-{tensor}'] = weight

            # Combine the grammars
            combined_grammar = {
                'Tensor': {**normal_updated_grammar['Tensor'], **uminus_updated_grammar},
                'Op': normal_updated_grammar['Op']  # Ops remain the same
            }
        else:
            combined_grammar = normal_updated_grammar

        # grammar_scaler = GrammarScaler(self.nn_grammar, combined_grammar)
        # combined_grammar = grammar_scaler.scaled_grammar
        return combined_grammar
    

# ------------------------------------------------------------------------------
# GrammarScaler Class
# ------------------------------------------------------------------------------
class GrammarScaler:
    """
    Optionally used to rescale grammar weights according to 
    certain heuristics or references. This is demonstrated 
    but not fully invoked in the code.

    Attributes:
        nn_wcfg (dict): The original neural network rule usage dictionary.
        updated_grammar (dict): The partially updated grammar so far.
        scaled_grammar (dict): The final grammar after applying scaling logic.
    """

    def __init__(self, nn_wcfg, updated_grammar):
        """
        Initialize the GrammarScaler with both the reference WCFG data 
        and the updated grammar to be scaled.

        Parameters:
            nn_wcfg (dict): Original usage dictionary from neural network.
            updated_grammar (dict): Updated grammar to be potentially scaled.
        """
        self.nn_wcfg = nn_wcfg
        self.updated_grammar = updated_grammar
        self.scaled_grammar = self.scale_grammar()

    def parse_tensor(self, tensor):
        """
        Extract the base name and index list from a tensor string.

        Example:
          '-b(i,j)' -> base 'b', indices ['i','j'] 
                      (leading '-' indicates unary minus, stripped here).
          'b(i,j)'  -> base 'b', indices ['i','j'].
        
        Returns:
            (str, list): (base_tensor_name, [indices])
        """
        if tensor.startswith('-'):
            tensor = tensor[1:]
        match = re.match(r'^([a-zA-Z0-9]+)\(([^)]*)\)$', tensor)
        if match:
            base = match.group(1)
            indices = match.group(2).split(',')
            indices = [idx.strip() for idx in indices]
            return base, indices
        else:
            # No indices
            return tensor, []

    def scale_grammar(self):
        """
        Merge or rescale the grammar weights. If the updated grammar
        has a tensor already in nn_wcfg, we keep the nn_wcfg weight. 
        Otherwise, attempt to match it by base name and indices set.

        Returns:
            dict: A final scaled grammar.
        """
        scaled_grammar = {'Tensor': {}, 'Op': self.updated_grammar['Op']}

        # For each tensor in updated_grammar['Tensor']
        for tensor, weight in self.updated_grammar['Tensor'].items():
            if tensor in self.nn_wcfg['Tensor']:
                # Tensor was in nn_wcfg, keep its weight
                scaled_grammar['Tensor'][tensor] = self.nn_wcfg['Tensor'][tensor]
            else:
                # Tensor was not in nn_wcfg, need to assign weight
                # Extract base tensor and index letters
                base_tensor, indices = self.parse_tensor(tensor)
                if not indices:
                    # If no indices, assign default weight
                    scaled_grammar['Tensor'][tensor] = weight
                    continue
                index_set = set(indices)
                # Find all tensors in nn_wcfg with same base and same index letters
                candidate_weights = []
                for nn_tensor, nn_weight in self.nn_wcfg['Tensor'].items():
                    nn_base, nn_indices = self.parse_tensor(nn_tensor)
                    if nn_base == base_tensor and set(nn_indices) == index_set:
                        candidate_weights.append(nn_weight)
                if candidate_weights:
                    max_weight = max(candidate_weights)
                    scaled_grammar['Tensor'][tensor] = max_weight
                else:
                    # No matching tensors in nn_wcfg, assign default weight
                    scaled_grammar['Tensor'][tensor] = weight

        return scaled_grammar
    

if __name__ == "__main__":
    nn_wcfg = {'Tensor': {'b(i,j)': 9, 'c(i)': 9, 'd(j)': 9, 'b(i)': 1, 'c(j)': 1, 'd(i,j)': 1}, 'Op': {'*': 10, '+': 10}}

    orders = [2, 2, 1, 1]
    include_constants  = False
    order_lenght_match = True
    grammar_updater = GrammarUpdater(nn_wcfg, orders[1:], include_constants, order_lenght_match)
    updated_grammar = grammar_updater.update_grammar()

    print(updated_grammar)
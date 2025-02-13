import math
import heapq
import functools
from collections import defaultdict
import itertools
import os
import subprocess
import re  
import sympy as sp
import unittest

import jax.numpy as jnp
import export_to_JAX
import taco_program

from io_handler import IOHandler
from check import check, CheckingReturnCode
from candidate import Candidate
from build_wcfg import *
candidates_tried_td = 0



# ------------------------------------------------------------------------------
# Gamma correction functions
# ------------------------------------------------------------------------------
def gamma_correction(value, gamma):
    """
    Apply gamma correction to a given value.

    Parameters:
    value (int): The original numeric value to be corrected.
    gamma (float): The gamma value used for correction.

    Returns:
    int: The gamma-corrected value.
    """    
    return int((value / 10) ** gamma * 10)

def apply_gamma_correction(lst, gamma, g_switch):
    """
    Apply gamma correction to a list of values, with different modes of adjustment 
    based on g_switch.

    Parameters:
    lst (list of int): List of numeric values to correct.
    gamma (float): Gamma value used for correction.
    g_switch (int): Mode switch for gamma application:
        1 -> Correct (value+1) 
        2 -> Correct (value)
        Otherwise -> Correct (value) and then add 1

    Returns:
    list of int: The list of gamma-corrected values.
    """
    if g_switch == 1:
        return [gamma_correction(value + 1, gamma) for value in lst]
    elif g_switch == 2:
        return [gamma_correction(value, gamma) for value in lst]
    else:
        return [gamma_correction(value, gamma) + 1 for value in lst]


# ------------------------------------------------------------------------------
# ExpressionNode class
# ------------------------------------------------------------------------------
class ExpressionNode:
    """
    Represents a node in an expression tree. The node can be either a terminal 
    (e.g., a tensor or constant) or non-terminal (an operator or placeholder).
    """

    def __init__(self, symbol, grammar, children=None, depth=1):
        """
        Initialize an ExpressionNode.

        Parameters:
        symbol (str): The symbol this node represents (e.g., 'Tensor', 'b(i,j)', '+').
        grammar (dict): The grammar dict used for expansions of non-terminals.
        children (list of ExpressionNode): Child nodes in this expression. Defaults to None.
        depth (int): The depth of this node within the expression tree. Defaults to 1.
        """
        self.symbol = symbol  
        self.grammar = grammar
        self.children = children or []
        self.depth = depth  

    def is_terminal(self):
        """
        Check if this node is a terminal (i.e., cannot be expanded further).

        Returns:
        bool: True if the node is terminal, False otherwise.
        """
        return self.symbol not in self.grammar

    def is_non_terminal(self):
        """
        Check if this node is a non-terminal (i.e., can be expanded based on the grammar).

        Returns:
        bool: True if the node is non-terminal, False otherwise.
        """
        return self.symbol in self.grammar

    def __str__(self):
        """
        String representation of the expression node. If it's an operator 
        node like '+', '-', '*', or '/', it prints in infix form. Otherwise, 
        it prints functional form like 'symbol(child1, child2)'.

        Returns:
        str: The string representation of this node.
        """
        if not self.children:
            return self.symbol
        elif self.symbol in ['+', '-', '*', '/']:
            left_str = str(self.children[0])
            right_str = str(self.children[1])
            return f"({left_str} {self.symbol} {right_str})"
        else:
            children_str = ', '.join(str(child) for child in self.children)
            return f"{self.symbol}({children_str})"

    def __repr__(self):
        """Alias to __str__ for debugging prints."""
        return self.__str__()

    def get_symbols(self):
        """
        Collect all symbols from this node's subtree.

        Returns:
        list of str: All symbols (including child symbols) found in the subtree.
        """
        if self.children:
            symbols = []
            for child in self.children:
                symbols.extend(child.get_symbols())
            return symbols
        else:
            return [self.symbol]

    def contains_non_terminal(self):
        """
        Check if this node or any of its descendants contains a non-terminal.

        Returns:
        bool: True if a non-terminal is found, False otherwise.
        """
        if self.is_non_terminal():
            return True
        for child in self.children:
            if child.contains_non_terminal():
                return True
        return False

    def get_depth(self):
        """
        Compute the maximum depth of the subtree rooted at this node.

        Returns:
        int: The maximum depth within this node's subtree.
        """
        if not self.children:
            return self.depth
        else:
            return max(child.get_depth() for child in self.children)


# ------------------------------------------------------------------------------
# PriorityQueue class
# ------------------------------------------------------------------------------
class PriorityQueue:
    """
    A priority queue implementation using a heap, allowing insertion and retrieval 
    of items based on priority (lowest priority value is popped first).
    """

    def __init__(self):
        """
        Initialize the priority queue. The queue uses a min-heap where each entry is 
        structured as [priority, count, item]. The 'count' is used to break ties 
        in priority.
        """
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def empty(self):
        """
        Check if the priority queue is empty.

        Returns:
        bool: True if the queue is empty, False otherwise.
        """
        return not self.heap

    def put(self, item, priority):
        """
        Insert an item with a given priority. If the item already exists in 
        the queue, it is marked as removed and reinserted with the new priority.

        Parameters:
        item (tuple): The item to be stored in the queue.
        priority (float): The priority value for the item (smaller means higher priority).
        """
        item_key = str(item[0])  
        if item_key in self.entry_finder:
            self.remove_task(item_key)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item_key] = entry
        heapq.heappush(self.heap, entry)

    def remove_task(self, item_key):
        """
        Mark an existing item as REMOVED in the queue. This does not completely remove 
        the item from memory until it is popped from the heap.

        Parameters:
        item_key (str): The key representing an item in the queue.
        """
        entry = self.entry_finder.pop(item_key)
        entry[-1] = self.REMOVED

    def get(self):
        """
        Pop the highest-priority item from the queue (lowest priority value).

        Returns:
        (priority, item): A tuple containing the popped item's priority 
                          and the item itself.

        Raises:
        KeyError: If the queue is empty.
        """
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            item_key = str(item[0])
            if item_key in self.entry_finder and self.entry_finder[item_key][-1] != self.REMOVED:
                del self.entry_finder[item_key]
                return priority, item
            elif item_key not in self.entry_finder:
                return priority, item
        raise KeyError('pop from an empty priority queue')


# ------------------------------------------------------------------------------
# td_aStarSynthesizer class
# ------------------------------------------------------------------------------
class td_aStarSynthesizer:
    """
    A top-down A* synthesizer that attempts to generate valid expression trees
    up to a maximum depth. It uses a weighted PCFG for expansions and 
    a cost function to guide the search.
    """

    def __init__(self, benchmark, lhs, orig_grammar, max_depth, io_set, order, grammar_style):
        """
        Initialize the top-down A* synthesizer.

        Parameters:
        benchmark (str): Name or identifier for the benchmark/test case.
        lhs (str): The left-hand side of the assignment of the final target.
        orig_grammar (dict): The original grammar dict from LLM.
        max_depth (int): The maximum tree depth allowed for generated expressions.
        io_set (list): The input-output set for evaluating candidate expressions.
        order (list): The sequence of tensor orders.
        grammar_style (str): The style of grammar being used (e.g., 'wcfg').
        """
        self.benchmark = benchmark
        self.lhs = lhs
        self.orig_grammar = orig_grammar
        self.grammar = self.convert_grammar()
        self.io_set = io_set
        self.max_depth = max_depth
        self.max_score = float('inf')
        self.stat_map = self._compute_stat_map()
        self.m = self.compute_m()
        self.m_sens = self.compute_m_sens()
        self.order = order
        self.grammar_style = grammar_style

    def convert_grammar(self):
        """
        Convert the original grammar into a format where both 'Tensor' expansions and 
        'Op' expansions reside under a single key 'Tensor'.

        Returns:
        dict: The updated grammar dict.
        """
        updated_grammar = {'Tensor': {}}
        for key, value in self.orig_grammar['Tensor'].items():
            updated_grammar['Tensor'][(key,)] = value        
        for op, value in self.orig_grammar['Op'].items():
            updated_grammar['Tensor'][('Tensor', op, 'Tensor')] = value
        return updated_grammar

    def _compute_stat_map(self):
        """
        Build a map of production probabilities for each non-terminal under different 
        contexts.

        Returns:
        defaultdict: A dictionary mapping context -> { (NT, production): probability }.
        """
        stat_map = defaultdict(dict)
        for nt, productions in self.grammar.items():
            total = sum(productions.values())
            for prod, weight in productions.items():
                context = ''
                stat_map[context][(nt, prod)] = weight / total
        return stat_map

    def compute_m(self):
        """
        Compute the minimum log probability cost for each non-terminal by 
        iterating until convergence. This is used as a base heuristic.

        Returns:
        dict: A dictionary mapping NT -> minimum cost.
        """
        m = {}
        rules = self.grammar
        for nt in rules:
            m[nt] = self.max_score
        changed = True
        while changed:
            changed = False
            for nt in rules:
                min_value = self.max_score
                for production in rules[nt]:
                    symbols = production
                    prod_prob = max(
                        [self.stat_map.get(ctx, {}).get((nt, production), 0.001) for ctx in self.stat_map],
                        default=0.001)
                    value = -math.log2(prod_prob)
                    for sym in symbols:
                        if sym in m:
                            value += m[sym]
                    if value < min_value:
                        min_value = value
                if abs(m[nt] - min_value) > 0.01:
                    m[nt] = min_value
                    changed = True
        return m

    def heuristic(self, node):
        """
        Compute a simple heuristic (estimated cost) of an expression node 
        based on the precomputed 'm' cost for non-terminals.

        Parameters:
        node (ExpressionNode): The expression node to evaluate.

        Returns:
        float: The heuristic cost estimate.
        """
        if node.is_terminal():
            return 0
        else:
            h_value = self.m.get(node.symbol, self.max_score)
            for child in node.children:
                h_value += self.heuristic(child)
            return h_value

    def compute_m_sens(self):
        """
        Compute context-sensitive version of 'm', which considers expansions 
        under different contexts. Currently, the context is always '', but 
        this is set up for potential future contexts.

        Returns:
        defaultdict(dict): A two-level dict mapping NT -> context -> cost.
        """
        m_sens = defaultdict(dict)
        rules = self.grammar

        for nt in rules:
            for context in self.stat_map:
                min_value = self.max_score
                for production in rules[nt]:
                    symbols = production
                    prod_prob = self.stat_map[context].get((nt, production), 0.001)
                    value = -math.log2(prod_prob)
                    for sym in symbols:
                        if sym in self.m:
                            value += self.m[sym]
                    if value < min_value:
                        min_value = value
                m_sens[nt][context] = min_value
        return m_sens

    def heuristic_sens(self, node, context):
        """
        Compute a context-sensitive heuristic cost for a node.

        Parameters:
        node (ExpressionNode): The node to evaluate.
        context (str): The context string.

        Returns:
        float: The context-sensitive heuristic cost.
        """
        if node.is_terminal():
            h_value = 0
        else:
            nt = node.symbol
            context_prob = self.m_sens.get(nt, {}).get(context, -math.log2(0.001))
            h_value = context_prob
            for child in node.children:
                h_value += self.heuristic_sens(child, context)
        return h_value

    def expansion_cost(self, nt, production, context):
        """
        Compute the negative log probability of expanding a non-terminal into
        a given production under the specified context.

        Parameters:
        nt (str): The non-terminal symbol being expanded.
        production (tuple): The production (e.g., ('Tensor', '+', 'Tensor')).
        context (str): The context (currently '').

        Returns:
        float: The expansion cost (negative log probability).
        """
        prob = self.stat_map[context].get((nt, production), 0.001)
        return -math.log2(prob)

    def is_fully_expanded(self, node):
        """
        Check if the entire subtree from this node has no non-terminals.

        Parameters:
        node (ExpressionNode): The root of the subtree to check.

        Returns:
        bool: True if no non-terminals remain, False otherwise.
        """
        return not node.contains_non_terminal()

    def is_alphabetical(self, expr_str):
        """
        Check if an expression is in alphabetical order in terms of the 
        tensor names it uses. This helps filter out invalid or unwanted 
        expressions.

        Parameters:
        expr_str (str): The string representation of the expression.

        Returns:
        bool: True if alphabetical order is maintained, False otherwise.
        """
        if 'e' in expr_str and 'e(' not in expr_str and 'Tensor' not in expr_str:
            return False
        if expr_str.count('Cons') > 1:
            return False
        expr_str = expr_str.replace(' ', '')

        tensor_names = [re.escape(t) for t in self.orig_grammar['Tensor'].keys()]
        tensor_pattern_str = r'(' + '|'.join(tensor_names) + r')'
        tensor_pattern = re.compile(tensor_pattern_str)

        def extract_tensors(term):
            tensors = tensor_pattern.findall(term)
            return tensors

        def tensors_in_order(tensors):
            for i in range(len(tensors) - 1):
                if tensors[i] > tensors[i + 1]:
                    return False
            return True

        terms = re.split(r'(?<!\w)[*/](?!\w)', expr_str)
        for term in terms:
            operands = re.split(r'(?<!\w)[+-](?!\w)', term)
            tensors_in_operands = []
            for operand in operands:
                tensors = extract_tensors(operand)
                tensors_in_operands.extend(tensors)
            if not tensors_in_order(tensors_in_operands):
                return False
        return True
    
    def op_used(self, expression):
        """
        Calculate the proportion of distinct operators used in the expression 
        relative to the total set of available operators in the grammar.

        Parameters:
        expression (str): The expression string.

        Returns:
        float: The fraction of operators used, or 1.0 if there is at most 
               one operator in the grammar.
        """
        total_ops = len(self.orig_grammar.get('Op', {}))
        if total_ops == 0 or total_ops == 1:
            return 1
        operator_set = set(self.orig_grammar.get('Op', {}).keys())
        expression = expression.replace(' ', '')
        op_pattern = '[' + re.escape(''.join(operator_set)) + ']'
        operators_in_expression = re.findall(op_pattern, expression)
        used_ops = set(operators_in_expression)
        proportion = len(used_ops) / total_ops if total_ops > 0 else 0
        return proportion 
    
    def has_identical_asd(self, expr_str):
        """
        Check for trivial or identical additive/subtractive or multiplicative terms 
        in the expression using sympy. It searches for patterns such as x + x, 
        x - x, etc.

        Parameters:
        expr_str (str): The expression string.

        Returns:
        bool: True if symmetrical or trivial pairs are found, False otherwise.
        """
        try:
            # # Define symbols and functions
            # idx = sp.symbols('i j k l')
            # b, c, d, e, f = [sp.Function(name) for name in ['b', 'c', 'd', 'e', 'f']]
            # Cons = sp.Symbol('Cons')

            expr = sp.sympify(expr_str, evaluate=False)

            def check_for_identical_terms(expr):
                if isinstance(expr, sp.Add):
                    terms = expr.args
                    term_set = set(terms)
                    for t1 in term_set:
                        if -t1 in term_set and t1 != 0:
                            return True
                    term_counts = {}
                    for t in terms:
                        term_counts[t] = term_counts.get(t, 0) + 1
                        if term_counts[t] >= 2:
                            return True
                    for term in terms:
                        if check_for_identical_terms(term):
                            return True
                elif isinstance(expr, sp.Mul):
                    factors = expr.args
                    for i in range(len(factors)):
                        for j in range(i + 1, len(factors)):
                            fi = factors[i]
                            fj = factors[j]
                            if isinstance(fi, sp.Pow) and isinstance(fj, sp.Pow):
                                if fi.base == fj.base and fi.exp == -fj.exp and fi.exp != 0:
                                    return True
                            elif isinstance(fi, sp.Pow):
                                if fi.base == fj and fi.exp == -1:
                                    return True
                            elif isinstance(fj, sp.Pow):
                                if fj.base == fi and fj.exp == -1:
                                    return True
                    for factor in factors:
                        if check_for_identical_terms(factor):
                            return True
                elif isinstance(expr, sp.Pow):
                    base = expr.base
                    exp = expr.exp
                    if exp == -1:
                        if check_for_identical_terms(base):
                            return True
                    else:
                        if check_for_identical_terms(base):
                            return True
                        if check_for_identical_terms(exp):
                            return True
                elif isinstance(expr, sp.Function):
                    for arg in expr.args:
                        if check_for_identical_terms(arg):
                            return True
                elif isinstance(expr, sp.Symbol):
                    return False
                else:
                    return False
                return False

            return check_for_identical_terms(expr)

        except sp.SympifyError:
            print("Invalid expression")
            return False
        
    def calculate_penalty(self, expr_str, penalty_weight, expected_length):
        """
        Calculate a penalty for the expression based on deviation from expected 
        length and other specific conditions (like using 'Cons').

        Parameters:
        expr_str (str): The expression string.
        penalty_weight (int): A base weight used for penalizing certain conditions.
        expected_length (int): The expected number of 'terms' in the expression.

        Returns:
        int: The total penalty score for the expression.
        """
        penalty = 0
        if expr_str.startswith('(') and expr_str.endswith(')'):
            expr_str = expr_str[1:-1]
        expr_length = self.length_of_expression(expr_str)
        if expr_length != expected_length and not expr_str.startswith('-'):
            penalty += penalty_weight*10
        if 'Cons' in self.orig_grammar['Tensor'] and len(self.order) > 3:
            if 'Cons' not in expr_str or expr_str.count('i') <= 1:
                penalty += penalty_weight
        return penalty
        
    def generate(self):
        """
        Perform an A* search over the space of expressions up to the maximum depth 
        specified. The search stops if it finds a valid expression that satisfies 
        the 'check' function, or if all expansions are exhausted.

        Returns:
        (Candidate, int) or ('no solution', int): 
            - Candidate instance and the count of candidates tried if successful
            - 'no solution' and count of candidates tried if no valid expression is found.
        """
        frontier = PriorityQueue()
        start_node = ExpressionNode('Tensor', self.grammar)
        start_item = (start_node, (), {}, False)  # Include penalized flag as False
        start_priority = self.heuristic(start_node)
        frontier.put(start_item, start_priority)

        global candidates_tried_td
        flag = False
        if any(key[0].startswith('b(') for key in self.grammar['Tensor']):
            flag = True
        expected_length = len(self.order[1:])

        penalty_weight = 1000  

        while not frontier.empty():
            priority, (current_node, derivation, tensor_mapping, penalized) = frontier.get()

            expr_str = str(current_node)
            check_first_tensor = expr_str.replace('(', '').replace(')', '').replace(' ', '')
            if flag:
                if check_first_tensor[0] not in ['b', 'C', 'T', '-']:
                    continue
            else:
                if check_first_tensor[0] not in ['b', 'c', 'C', 'T', '-']:
                    continue

            current_depth = current_node.get_depth()
            if current_depth > self.max_depth:
                continue
            if not self.is_alphabetical(expr_str):
                continue
            if self.is_fully_expanded(current_node):
                if self.has_identical_asd(expr_str):
                    continue
                if self.op_used(expr_str) < 1 / 2 and self.grammar_style == 'wcfg':
                    continue
                if not penalized:
                    penalty = self.calculate_penalty(expr_str, penalty_weight, expected_length)
                    new_priority = priority + penalty
                    frontier.put((current_node, derivation, tensor_mapping, True), new_priority)
                    continue  
                else:
                    candidate = Candidate(self.lhs, expr_str)
                    candidates_tried_td += 1
                    check_return_code = check(self.benchmark, candidate, self.io_set, candidates_tried_td)
                    if check_return_code == CheckingReturnCode.SUCCESS:
                        return candidate, candidates_tried_td
                    continue  
            nodes_to_expand = self.find_non_terminal_nodes(current_node)
            if not nodes_to_expand:
                continue
            node_to_expand = nodes_to_expand[0]
            nt = node_to_expand.symbol

            context = ''

            # Expand the non-terminal
            for production in self.grammar[nt]:
                production_valid = True
                new_tensor_mapping = tensor_mapping.copy()
                if len(production) == 1:
                    sym = production[0]
                    if sym in self.grammar:
                        pass  
                    else:
                        tensor_match = re.match(r'([a-zA-Z])\(', sym)
                        if tensor_match:
                            tensor_letter = tensor_match.group(1)
                            if tensor_letter in tensor_mapping:
                                if tensor_mapping[tensor_letter] != sym:
                                    production_valid = False
                            else:
                                new_tensor_mapping[tensor_letter] = sym
                if not production_valid:
                    continue 
                if len(production) == 1:
                    sym = production[0]
                    new_subtree = ExpressionNode(sym, self.grammar, depth=node_to_expand.depth + 1)
                else:
                    operator = production[1]
                    left_sym = production[0]
                    right_sym = production[2]
                    left_node = ExpressionNode(left_sym, self.grammar, depth=node_to_expand.depth + 1)
                    right_node = ExpressionNode(right_sym, self.grammar, depth=node_to_expand.depth + 1)
                    new_subtree = ExpressionNode(operator, self.grammar, [left_node, right_node], depth=node_to_expand.depth)

                new_node = self.replace_node(current_node, node_to_expand, new_subtree)

                current_depth = new_node.get_depth()
                if current_depth > self.max_depth:
                    continue  
                new_derivation = derivation + (expr_str,)

                expand_cost = self.expansion_cost(nt, production, context)
                heuristic_cost = self.heuristic_sens(new_node, context)

                new_expr_str = str(new_node)
                penalty = self.calculate_penalty(new_expr_str, penalty_weight, expected_length)

                new_priority = priority + expand_cost + heuristic_cost + penalty

                frontier.put((new_node, new_derivation, new_tensor_mapping, False), new_priority)

        return 'no solution', candidates_tried_td
    
    def length_of_expression(self, expr_str):
        """
        Estimate the "length" of the expression by splitting on arithmetic operators 
        (+, -, *, /).

        Parameters:
        expr_str (str): The expression string.

        Returns:
        int: The number of terms found in the expression after splitting on arithmetic
             operators.
        """
        terms = re.split(r'[+\-*/]', expr_str)
        return len(terms)

    def find_non_terminal_nodes(self, node):
        """
        Find the next non-terminal node in the subtree. This method returns
        only the first encountered non-terminal list for immediate expansion
        in a left-to-right manner.

        Parameters:
        node (ExpressionNode): The current node.

        Returns:
        list of ExpressionNode: A list containing the first non-terminal node found, 
                                or an empty list if none found.
        """
        nodes = []
        if node.is_non_terminal():
            nodes.append(node)
            return nodes 
        for child in node.children:
            child_nodes = self.find_non_terminal_nodes(child)
            if child_nodes:
                return child_nodes  
        return nodes

    def replace_node(self, current_node, target_node, new_subtree):
        """
        Recursively traverse the current_node subtree, and replace 'target_node' 
        with 'new_subtree'.

        Parameters:
        current_node (ExpressionNode): The root node where replacement will happen.
        target_node (ExpressionNode): The node to be replaced.
        new_subtree (ExpressionNode): The new node to insert in place of target_node.

        Returns:
        ExpressionNode: The modified expression tree with the replacement.
        """
        if current_node is target_node:
            return new_subtree
        elif current_node.children:
            new_child_nodes = []
            for child in current_node.children:
                replaced_child = self.replace_node(child, target_node, new_subtree)
                new_child_nodes.append(replaced_child)
            return ExpressionNode(current_node.symbol, self.grammar, new_child_nodes, depth=current_node.depth)
        else:
            return current_node

    def length_difference_after_expansion(self, node, production):
        """
        (Optional / Example) Compute how the expression length changes after 
        a certain expansion. Currently not used in the code directly.

        Parameters:
        node (ExpressionNode): The node being expanded.
        production (tuple): The production rule to apply.

        Returns:
        int: The difference in expression terms.
        """
        if len(production) == 1:
            return 1
        else:
            return sum(1 if sym not in self.grammar else 0 for sym in production)

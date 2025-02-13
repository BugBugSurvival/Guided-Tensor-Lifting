# Description: A* synthesizer for PCFGs
import math
import heapq
from check import check, CheckingReturnCode
from candidate import Candidate
from build_wcfg import *
import math
import heapq
import re  # Added for regex operations
from verify import *
from io_handler import IOHandler

candidates_tried_bu = 0

class PriorityQueue:
    """
    A priority queue implementation using Python's heapq.

    This class stores items along with their priorities. The item with
    the smallest priority value is popped first (min-heap behavior).
    """
    def __init__(self):
        """
        Initialize an empty priority queue.
        """
        self.heap = []
        self.elements = set()
        self.nelem = 0

    def empty(self):
        """
        Check if the priority queue is empty.

        Returns:
            bool: True if empty, False otherwise.
        """
        return self.nelem == 0

    def get(self):
        """
        Pop (remove and return) the item with the lowest priority value.

        Returns:
            tuple: A tuple (item, priority).
        
        Raises:
            IndexError: If the priority queue is empty.
        """
        pri, d = heapq.heappop(self.heap)
        self.nelem -= 1
        self.elements.remove(d)
        return d, pri

    def put(self, item, priority):
        """
        Insert an item into the priority queue with a given priority.
        If the item is already in the queue, do not insert it again.

        Parameters:
            item (object): The item to insert.
            priority (float): The priority value (lower is higher priority).
        """
        if item in self.elements:
            return
        heapq.heappush(self.heap, (priority, item))
        self.nelem += 1
        self.elements.add(item)

# ------------------------------------------------------------------------------
# bu_aStarSynthesizer class
# ------------------------------------------------------------------------------
class bu_aStarSynthesizer:
    """
    Bottom-up A* synthesizer using a PCFG. It attempts to build expressions in 
    a stepwise manner up to the desired dimensional order (tensor_order_list).
    """

    def __init__(self, benchmark, grammar, tensor_order_list, io_set, grammar_style):
        """
        Initialize the bottom-up A* synthesizer.

        Parameters:
            benchmark (str): A name/identifier for the benchmark problem.
            grammar (dict): The grammar dictionary containing 'Tensor' and 'Op' expansions.
            tensor_order_list (list): The sequence of tensor orders to synthesize.
            io_set (list): The input/output set for validation (test cases).
            grammar_style (str): Type of grammar ("wcfg", etc.).
        """
        self.benchmark = benchmark
        self.orig_grammar = grammar
        if len(self.orig_grammar.get('Op', {})) == 0 and len(tensor_order_list) != 2:
            tensor_order_list = tensor_order_list[:1] + tensor_order_list[2:]
        self.grammar = self.convert_grammar(grammar, tensor_order_list)
        print(self.grammar)
        self.tensor_order_list = tensor_order_list[1:]
        self.prog = tensor_order_list[0]
        self.io_set = io_set
        self.frontier = PriorityQueue()
        self.enumerated_exps = []
        self.m = self.compute_m()
        self.m_sens = self.compute_m_sens()
        self.init_priority_queue()
        self.grammar_style = grammar_style

    def convert_grammar(self, original_grammar, tensor_order_list):
        """
        Convert the original grammar into a new structure that separates
        Tensor expansions by their order and normalizes their weights.

        Parameters:
            original_grammar (dict): Original grammar containing 'Tensor' and 'Op'.
            tensor_order_list (list): The list of tensor orders.

        Returns:
            dict: A transformed grammar dictionary where:
                  - 'Tensor': {order : {tensor_name : probability}}
                  - 'Op': {operator : probability}
        """
        new_grammar = {'Tensor': {}, 'Op': original_grammar['Op']}
        for order in tensor_order_list:
            new_grammar['Tensor'][order] = {}

        def determine_order(tensor_name):
            """
            Determine the order (number of indices) of a given tensor name.
            Example: b(i, j) -> 2
            """
            if '(' in tensor_name and ')' in tensor_name:
                indices = tensor_name.split('(')[1].split(')')[0].split(',')
                return len(indices)
            return 0

        for tensor, count in original_grammar['Tensor'].items():
            order = determine_order(tensor)
            if order in new_grammar['Tensor']:
                new_grammar['Tensor'][order][tensor] = count

        epsilon = 1e-10
        for key, orders in new_grammar['Tensor'].items():
            total = sum(orders.values())
            for prod in orders:
                new_grammar['Tensor'][key][prod] = (orders[prod] / total) + epsilon

        filtered_ops = {op: weight for op, weight in new_grammar['Op'].items() if weight > 0}
        new_grammar['Op'] = filtered_ops
        total_op = sum(new_grammar['Op'].values())
        for op in new_grammar['Op']:
            new_grammar['Op'][op] = (new_grammar['Op'][op] / total_op) + epsilon
        return new_grammar

    def init_priority_queue(self):
        """
        Initialize the priority queue with all possible tensors of the first order
        in the tensor_order_list. This acts as the starting point for the synthesis.
        """
        initial_order = self.tensor_order_list[0]
        if initial_order in self.grammar['Tensor']:
            for tensor in self.grammar['Tensor'][initial_order]:
                self.frontier.put(((tensor,), 1), -math.log2(self.grammar['Tensor'][initial_order][tensor]))

    def compute_m(self):
        """
        Compute a basic heuristic cost for each tensor order using negative log probabilities.

        Returns:
            dict: A dictionary { order: cost } storing the minimal cost.
        """
        rules = self.grammar['Tensor']
        m = {}
        stat_map = {key: {k: v for k, v in rules[key].items()} for key in rules}
        for nt, prods in rules.items():
            new_value = float('inf')
            for prod, prob in prods.items():
                topsymb = prod
                max_prob = max([topsymb_to_prob.get(topsymb, 0.001) for ctx, topsymb_to_prob in stat_map.items()], default=0.001)
                new_value = min(new_value, -math.log2(max_prob))
            m[nt] = new_value

        def F():
            updated = False
            for nt, prods in rules.items():
                old_value = m.get(nt, float('inf'))
                new_value = old_value
                for prod, prob in prods.items():
                    topsymb = prod
                    max_prob = max([topsymb_to_prob.get(topsymb, 0.001) for ctx, topsymb_to_prob in stat_map.items()], default=0.001)
                    new_value = min(new_value, -math.log2(max_prob))
                updated = updated or (abs(old_value - new_value) > 0.01)
                m[nt] = new_value

            return updated

        updated = True
        while updated:
            updated = F()
        return m

    def compute_m_sens(self):
        stat_map = {key: 1.0 for key in self.grammar['Tensor']}
        rules = self.grammar['Tensor']
        m_sens = {}
        for nt, prods in rules.items():
            for ctx, topsymb_to_prob in stat_map.items():
                min_logprob = float('inf')
                for prod, prob in prods.items():
                    min_logprob = min(min_logprob, -math.log2(prob))
                m_sens[nt] = min_logprob

        return m_sens

    def heuristic(self, expr):
        """
        Compute a heuristic for an in-progress expression. 

        Parameters:
            expr (tuple of str): The current expression in tuple form.

        Returns:
            float: The heuristic score estimating additional cost to complete the expression.
        """
        next_nts = [self.tensor_order_list[len(expr) // 2]]
        score = sum(self.m.get(nt, float('inf')) for nt in next_nts)
        return score

    def is_alphabetical(self, expr):
        """
        Check if an expression's tensor parts appear in non-decreasing alphabetical order.
        
        Parameters:
            expr (tuple of str): The current expression tuple (tensors/operators).

        Returns:
            bool: True if alphabetical, False otherwise.
        """
        tensors = [t for t in expr if not any(op in t for op in self.grammar['Op'])]
        return all(tensors[i] <= tensors[i + 1] for i in range(len(tensors) - 1))
    
    def op_used(self, expression):
        """
        Calculate the fraction of available operators that appear in the final expression.
        A fraction of zero indicates no operators used; if no operators are in the grammar,
        it returns 1 to avoid division by zero.

        Parameters:
            expression (str): A space-separated expression string.

        Returns:
            float: Fraction of distinct operators used in the expression.
        """
        total_ops = len(self.orig_grammar.get('Op', {}))
        if total_ops == 0:
            return 1
        operator_set = set(self.orig_grammar.get('Op', {}).keys())
        expression = expression.replace(' ', '')
        op_pattern = '[' + re.escape(''.join(operator_set)) + ']'
        operators_in_expression = re.findall(op_pattern, expression)
        used_ops = set(operators_in_expression)
        if len(used_ops) == 0:
            return 1
        proportion = len(used_ops) / total_ops if total_ops > 0 else 0
        return proportion 
    
    def generate(self):
        """
        Perform the main bottom-up A* synthesis. The search proceeds by adding
        partial expressions to expression tails until the desired length of
        tensors is reached. Valid expressions are checked against the I/O set.

        Returns:
            tuple: (Candidate, int) if a correct expression is found,
                   or ('no solution', int) if no valid expression is produced.
        """
        global candidates_tried_bu
        non_alpha_penalty = 100  
        lhs = get_valid_indexations('a', self.prog)[0]
        len_order = len(self.tensor_order_list)
        while not self.frontier.empty():
            (current_expr, tensor_count), current_score = self.frontier.get()
            if 'e' in current_expr and 'e(' not in current_expr:
                continue
            if tensor_count == len_order:
                #self.enumerated_exps.append((" ".join(current_expr), current_score))
                rhs = " ".join(current_expr)
                if self.op_used(rhs) < 1/2 and self.grammar_style == 'wcfg':
                    continue
                candidate = Candidate(lhs, rhs)
                candidates_tried_bu += 1
                check_return_code = check(self.benchmark, candidate, self.io_set, candidates_tried_bu)
                if check_return_code == CheckingReturnCode.SUCCESS:
                    return candidate, candidates_tried_bu
            elif tensor_count < len_order:
                next_order = self.tensor_order_list[tensor_count]
                #last_tensor = current_expr[-1] if tensor_count > 0 else ""
                for op in self.grammar['Op']:
                    for tensor in sorted(self.grammar['Tensor'][next_order]):
                        new_expr = current_expr + (op, tensor)
                        new_score = current_score + -math.log2(self.grammar['Op'][op]) + -math.log2(self.grammar['Tensor'][next_order][tensor])
                        if not self.is_alphabetical(new_expr):
                            new_score += non_alpha_penalty
                        self.frontier.put((new_expr, tensor_count + 1), new_score + self.heuristic(new_expr))

        return 'no solution', candidates_tried_bu

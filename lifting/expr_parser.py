import ply.lex as lex
import ply.yacc as yacc
class ExprParser:
    """
    ExprParser is a class that uses PLY (Python Lex-Yacc) to parse 
    simple tensor expressions of the form:
    
        <tensor> = <expression>

    Where <expression> may involve:
      - Tensors with optional indices (e.g., b(i,j))
      - Constants (digits)
      - Operators (+, -, *, /)
      - Unary minus
      - Parentheses for grouping

    The parser also tracks usage of "Tensors" and "Operators" in 
    a dictionary attribute 'rule_usage'. This can be used downstream 
    to see how many times a particular tensor or operator was used.

    Attributes:
        lexer (ply.lex.lex): The lexer instance used to tokenize input strings.
        parser (ply.yacc.yacc): The parser instance used to build and parse expressions.
        rule_usage (dict): Tracks occurrences of Tensors and Operators, e.g.,
                           {
                             'Tensor': {'b(i,j)': 2, 'Cons': 3},
                             'Op': {'+': 1, '*': 2}
                           }
    """

    tokens = (
        'ID', 'CONSTANT',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
        'LPAREN', 'RPAREN',
        'COMMA', 'EQUAL'
    )

    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_COMMA = r','
    t_EQUAL = r'='
    
    # Constructor
    def __init__(self):
        """
        Constructor for ExprParser. Initializes the PLY lexer and parser,
        and sets up an empty rule_usage dictionary.
        """
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self)
        self.rule_usage = {'Tensor': {}, 'Op': {}}

    def t_CONSTANT(self, t):
        r'\d+'
        t.type = 'CONSTANT'
        return t

    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = 'ID'
        return t

    
    t_ignore = " \t\n"

    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)

    precedence = (
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('right', 'UMINUS'),  # Add precedence for unary minus
    )

    def p_program(self, p):
        'program : tensor EQUAL expr'
        p[0] = {
            'lhs': p[1],  # Store LHS as it is
            'rhs': p[3]   # Store RHS as parsed tuples
        }

    def p_tensor_with_index(self, p):
        'tensor : ID LPAREN index_expr RPAREN'
        p[0] = f"{p[1]}({p[3]})"

    def p_tensor(self, p):
        'tensor : ID'
        p[0] = p[1]

    def p_uminus_tensor(self, p):
        "uminusTensor : MINUS tensor"
        p[0] = f"-{p[2]}"

    def p_expr_uminus(self, p):
        "expr : MINUS expr %prec UMINUS"
        p[0] = ('-', p[2])  # Represent unary minus as a tuple

    def p_index_expr_single(self, p):
        'index_expr : index_tensor'
        p[0] = p[1]

    def p_index_expr_comma(self, p):
        'index_expr : index_tensor COMMA index_expr'
        p[0] = f"{p[1]},{p[3]}"

    def p_index_tensor(self, p):
        '''index_tensor : ID'''
        p[0] = p[1]

    def p_expr_op(self, p):
        '''expr : expr PLUS expr
                | expr MINUS expr
                | expr TIMES expr
                | expr DIVIDE expr'''
        p[0] = (p[1], p[2], p[3])  # Represent RHS as a tuple (operand, operator, operand)
        self.update_usage('Op', p[2])

    def p_expr_group(self, p):
        'expr : LPAREN expr RPAREN'
        p[0] = p[2]  # Return the expression inside parentheses

    def p_expr_constant_tensor(self, p):
        '''expr : CONSTANT
                | tensor
                | uminusTensor'''
        p[0] = p[1]
        if p.slice[1].type == 'CONSTANT':
            self.update_usage('Tensor', 'Cons')  # Track all constants under "Cons"
        elif p.slice[1].type == 'tensor':
            self.update_usage('Tensor', str(p[1]))  # Track tensor IDs properly
        elif p.slice[1].type == 'uminusTensor':
            self.update_usage('Tensor', str(p[1]))  # Track tensors with unary minus

    def p_error(self, p):
        print("Syntax error in input!")
        pass

    def update_usage(self, category, text):
        if category == 'Op':
            self.rule_usage['Op'][text] = self.rule_usage['Op'].get(text, 0) + 1
        elif category == 'Tensor':
            self.rule_usage['Tensor'][text] = self.rule_usage['Tensor'].get(text, 0) + 1
        elif category == 'uminusTensor':
            self.rule_usage['Tensor'][f"-{text}"] = self.rule_usage['Tensor'].get(f"-{text}", 0) + 1

    def parse(self, data):
        self.rule_usage = {'Tensor': {}, 'Op': {}}  # Reset for each parse
        return self.parser.parse(data)

# Example usage
if __name__ == "__main__":
    parser = ExprParser()
    
    input_expressions = [
        "a(i) = a(i) * c",
        "x(j,k) = y(k) / z(j)",
        "arr(i) = arr(i) + 1",
        "sum(i) = arr(i) - b(i)",
        "out(i) = alpha * x(i) + (1 - alpha) * y(i)"
    ]
    
    for expr in input_expressions:        
        print(f"Input Expression: {expr}")

        result = parser.parse(expr)
        if result:
            print(f"LHS: {result['lhs']}, RHS: {result['rhs']}")
        print("-" * 40)

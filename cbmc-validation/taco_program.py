import re

def _get_tensor_name(tensor):
  if tensor.startswith('Cons'):
    return tensor[1:] if tensor[0] == '-' else tensor
  else:
    return tensor[1] if tensor[0] == '-' else tensor[0]


def _get_tensor_order(tensor):
  """Returns the order of a given tensor leveraging its string representation."""
  return 0 if tensor.count('(') == 0 else tensor.count(',') + 1


def _get_tensor_indexing(tensor):
  if len(tensor) == 1 or re.match('Cons*.', tensor):
    return ''
  elif tensor[0] == '-':
    return tensor[2:].lstrip('(').rstrip(')')
  else:
    return tensor[1:].lstrip('(').rstrip(')')


def _is_negated(tensor):
  return True if tensor[0] == '-' else False


class Tensor():

  def __init__(self, name, order, indexing = '', negated = False):
    self.name = name
    self.order = order
    self.indexing = indexing
    self.negated = negated


  def __str__(self):
    tensor_as_str = '-' + self.name if self.negated else self.name
    tensor_as_str = tensor_as_str + ':' + self.indexing if self.order > 0 else tensor_as_str
    return tensor_as_str
  

  def __eq__(self, other):
    if isinstance(other, Tensor):
      return self.name == self.name


  def __hash__(self):
    return hash(self.name)
  

  def copy(self):
    return Tensor(self.name, self.order, self.indexing, self.negated)



class TACOProgram():
  
  def __init__(self, lhs_tensor, rhs_tensors = [], arithops = [], l_par = None, r_par = None):
    assert rhs_tensors, 'Program should have at least one tensors on the right-hand side.'
    if arithops:
      assert len(rhs_tensors) == len(arithops) + 1, 'The number of tensors should be the number of binary operators plus one.'
    
    self.lhs_tensor = lhs_tensor
    self.rhs_tensors = rhs_tensors
    self.tensor_names = [lhs_tensor.name] + [tensor.name for tensor in self.rhs_tensors]
    self.indexings = [lhs_tensor.indexing] + [tensor.indexing for tensor in self.rhs_tensors]
    self.arithops = arithops
    self.n_tensors = 1 + len(rhs_tensors)
    self.n_arithops = self.n_tensors - 2
    self.l_par = l_par
    self.r_par = r_par
  

  def __str__(self):
    program_as_str = str(self.lhs_tensor) + ' = '
    if self.n_tensors == 2:
      if 0 in self.l_par:
        return program_as_str + self.l_par[0] + str(self.rhs_tensors[0]) + self.r_par[0]
      else:
        return program_as_str + str(self.rhs_tensors[0])
    else:
      for i in range(self.n_arithops):
        if i in self.l_par:
          program_as_str += self.l_par[i]
        program_as_str += str(self.rhs_tensors[i])
        if i in self.r_par:
          program_as_str += self.r_par[i]
        program_as_str += ' ' + str(self.arithops[i]) + ' '

      program_as_str += str(self.rhs_tensors[self.n_tensors - 2])
      if self.n_tensors - 2 in self.r_par:
        program_as_str += self.r_par[self.n_tensors - 2]

      return program_as_str


  def __repr__(self):
    return self.__str__()
  

  def copy(self):
    lhs = self.lhs_tensor.copy()
    rhs = [t.copy() for t in self.rhs_tensors]
    return TACOProgram(lhs, rhs, self.arithops.copy(), self.mutation)


  def has_variable_tensor(self):
    return any(not tensor.name.isnumeric() for tensor in self.rhs_tensors)


  def get_reference_to_tensor(self, tensor_name):
    for tensor in self.rhs_tensors:
      if tensor.name == tensor_name:
        return tensor
  

  def has_tensor_negated(self):
    return any(tensor.negated for tensor in self.rhs_tensors)


  @classmethod
  def from_string(cls, program):
    tensors_regex = '^-?[b-z]'
    cons_placeholder_regex = '-?Cons*'
    tensors = []
    arithops = []
    l_par = dict()
    r_par = dict()

    lhs, rhs = program.split(' = ')
    lhs_tensor = Tensor(_get_tensor_name(lhs), _get_tensor_order(lhs), _get_tensor_indexing(lhs))
    
    rhs_pattern = r'(\(|\)|(?<!\w)-?[a-zA-Z_]\w*\([a-zA-Z_,\d]*\)|(?<!\w)-?[a-zA-Z_]\w*|(?<!\w)[+\*/-](?![a-zA-Z_]))'
    rhs_tensor_idx = 0
    for elem in re.findall(rhs_pattern, rhs):
      if re.match(tensors_regex, elem) or re.match(cons_placeholder_regex, elem):
        tensors.append(Tensor(_get_tensor_name(elem), _get_tensor_order(elem), _get_tensor_indexing(elem), _is_negated(elem)))
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
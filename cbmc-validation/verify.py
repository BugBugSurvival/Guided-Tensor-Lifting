import ast
from enum import Enum
import mlir
import re
import sys
import time

import mlir.dialect
import mlir.dialects

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import mlir_synth.synth as mlir_synth
from mlir_synth.ir import *
from mlir_synth.passmanager import *

import taco_program
import export_to_JAX


class OPT(Enum):
  RANK0_ARGS_TO_SCALARS = 1


class FuncVisitor(mlir.NodeVisitor):
  def __init__(self):
    super().__init__()
    self.args = []
    self.arg_types = []

  def visit_Function(self, node):
    self.args = [arg.name.value for arg in node.args]
    self.arg_types = [arg.type.dump() for arg in node.args]
    

def string_to_tuple_list(string):
  try:
    result = ast.literal_eval(string)
    if isinstance(result, list) and all(isinstance(i, tuple) and len(i) == 2 and all(isinstance(j, str) for j in i) for i in result):
      return result
    else:
      raise ValueError("Input does not match the expected format: list of string tuples")
  except (SyntaxError, ValueError) as e:
    raise ValueError(f"Invalid input format: {e}")


def get_original_arguments_with_regex(lowlevel_mlir_file):
  with open(lowlevel_mlir_file, "r") as mlir_f:
    mlir_code = mlir_f.read()

  pattern = r'%([a-zA-Z_]\w*):\s*([\w<>xf\d]+)'
  matches = re.findall(pattern, mlir_code)
    
  if not matches:
    raise ValueError("Function not found in MLIR code")

  names, types = zip(*matches) if matches else ([], [])
  return list(names), list(types)


def get_original_arguments_order(lowlevel_mlir_file):
  try:
    mod_ll = mlir.parse_path(lowlevel_mlir_file)
    visitor = FuncVisitor()
    visitor.visit(mod_ll)
    return visitor.args, visitor.arg_types
  except:
    # If we fail to parse the MLIR file we take the argument names using
    # a regex based approach
    return get_original_arguments_with_regex(lowlevel_mlir_file)


def get_arguments_rank(arg_types):
  arg_ranks = [type.count("x") for type in arg_types]
  options = []
  if any(rank == 0 for rank in arg_ranks):
    options = [OPT.RANK0_ARGS_TO_SCALARS]

  return arg_ranks, options


def check_equiv(mlir_low_level_str, mlir_high_level_str, options=[]):
  with Context():
    mlir_synth.register_dialects()
    mlir_synth.register_passes()

    # Low level
    mod_ll = Module.parse(mlir_low_level_str)
    pm_ll = PassManager.parse("change-sizes,return-output-arg")
    pm_ll.run(mod_ll)

    # - Get the function annotated with the "irsynth.original" attribute
    func_ll = mod_ll.body.operations[0]        

    # High level
    mod_hl = Module.parse(mlir_high_level_str)
    func_hl = mod_hl.body.operations[0]
    func_hl.attributes["irsynth.raised"] = UnitAttr.get()

    # Create modules with only the relevant functions
    mod_ll_only = Module.create(Location.unknown())
    ip = InsertionPoint(mod_ll_only.body)
    ip.insert(func_ll.detach_from_parent())

    mod_hl_only = Module.create(Location.unknown())
    ip = InsertionPoint(mod_hl_only.body)
    ip.insert(func_hl.detach_from_parent())

    # High level
    # - Lower to affine
    mlir_synth.lower_chlo_to_affine(mod_hl_only, False)
    # - Convert rank-0 memrefs to scalars
    if OPT.RANK0_ARGS_TO_SCALARS in options:
      pm = PassManager.parse("fold-memref-alias-ops,memref-rank0-to-scalar")
      pm.run(mod_hl_only)

    #print(mod_ll_only)
    #print(mod_hl_only)

    # Check equivalence
    return mlir_synth.check_validate(mod_ll_only, mod_hl_only)


def to_mlir_hlo(kernel, arg_ranks):
  arg_shapes = [tuple([3] * rank) for rank in arg_ranks]

  args = []
  for shape in arg_shapes:
    if len(shape) == 0:
      args.append(jnp.float64(1))
    else:
      args.append(jnp.ones(shape, dtype=jnp.float64))

  jax_kernel = jax.jit(kernel, backend="cpu", keep_unused=True)
  return str(jax_kernel.lower(*args).compiler_ir(dialect='mhlo'))


def verify(lowlevel_mlir_file, kernel, arg_ranks, options):
  with open(lowlevel_mlir_file, "r") as f:
    mlir_low_level_str = f.read()

  mlir_hlo_str = to_mlir_hlo(kernel, arg_ranks)
  return check_equiv(mlir_low_level_str, mlir_hlo_str, options)


if __name__ == "__main__":
  if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <path-to-original-mlir> <TACO program> <substitution>")
    exit(1)

  lowlevel_mlir_file = sys.argv[1]
  taco_prog = taco_program.TACOProgram.from_string(sys.argv[2])
  substitution = string_to_tuple_list(sys.argv[3])
  args_ordering, arg_types = get_original_arguments_order(lowlevel_mlir_file)
  print(f"Verifying {lowlevel_mlir_file}")
  lambda_vars, kernel_exp = export_to_JAX.export_TACO_to_JAX(taco_prog, args_ordering, substitution)
  kernel = eval(f"lambda {lambda_vars}: {kernel_exp}")
  arg_ranks, options = get_arguments_rank(arg_types)
  
  print(f"Dimensionality: {arg_ranks}")
  print(f"Options: {options}\n")
  
  start_v = time.time()
  verified = verify(lowlevel_mlir_file, kernel, arg_ranks, options)
  v_time = time.time() - start_v
  print(f"Equivalent: {verified}")
  print(f"Verification time: {v_time:.2f}")

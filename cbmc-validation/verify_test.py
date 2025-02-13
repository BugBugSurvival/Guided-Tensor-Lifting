import unittest

import jax.numpy as jnp
import export_to_JAX
import taco_program
import verify

class TestVerificationPipeline(unittest.TestCase):
  def test_cases(self):
    test_cases = [
      ("benchmark/tensor_algebra/blas/dot.mlir", "a = b(i) * c(i)", "[('a', 'b'), ('b', 'c')]"),
      ("benchmark/tensor_algebra/blas/gemv.mlir", "a(i) = b(i,j) * c(j)", "[('A', 'b'), ('x', 'c')]"),
      ("benchmark/tensor_algebra/blas/ger.mlir", "a(i,j) = b(i,j) + c(i) * d(j)", "[('a', 'b'), ('x', 'c'), ('y', 'd')]"),
      ("benchmark/tensor_algebra/darknet/gemm_nt.mlir", "a(i,j) = b(i,j) + c * d(i,k) * e(j,k)", "[('ALPHA', 'c'), ('A', 'd'), ('B', 'e'), ('C', 'b')]"),
      ("benchmark/tensor_algebra/darknet/ol_l2_cpu1.mlir", "a(i) = (b(i) - c(i)) * (b(i) - c(i))", "[('pred', 'c'), ('truth', 'b')]"),
      ("benchmark/tensor_algebra/dsp/vfill.mlir", "a(i) = b", "[('v', 'b')]"),
      ("benchmark/tensor_algebra/dsp/vneg.mlir", "a(i) = -b(i)", "[('arr', 'b')]"),
      ("benchmark/tensor_algebra/dsp/vrecip.mlir", "a(i) = Cons1 / b(i)", "[('arr', 'b'), ('1', 'Cons1')]"),
      ("benchmark/tensor_algebra/dsp/w_vec.mlir", "a(i) = b * c(i) + d(i)", "[('a', 'c'), ('b', 'd'), ('m', 'b')]"),
      ("benchmark/tensor_algebra/dspstone/n_real_updates.mlir", "a(i) = b(i) * c(i) + d(i)", "[('A', 'c'), ('B', 'b'), ('C', 'd')]"),
      ("benchmark/tensor_algebra/dspstone/matrix1.mlir", "a(i,j) = b(i,k) * c(j,k)", "[('A', 'b'), ('B', 'c')]"),
      ("benchmark/tensor_algebra/mathfu/diveq.mlir", "a(i) = b(i) / c(i)", "[('a', 'b'), ('b', 'c')]"),
      ("benchmark/tensor_algebra/mathfu/muleq_sca.mlir", "a(i) = b(i) * c", "[('a', 'b'), ('b', 'c')]"),
      ("benchmark/tensor_algebra/artificial/4_taco.mlir", "a(i,j) = b(i,k) * c(k,j)", "[('b', 'b'), ('c', 'c')]"),
      ("benchmark/tensor_algebra/artificial/5_taco.mlir", "a(i,j) = b(i)", "[('b', 'b')]"),
      ("benchmark/tensor_algebra/artificial/add.mlir", "a = b + c", "[('b', 'b'), ('c', 'c')]"),
      ("benchmark/tensor_algebra/artificial/array_prod_scalar.mlir", "a(i) = b(i) * c", "[('b', 'c'), ('c', 'b')]"),
    ]
      
    for mlir_file, taco_prog_str, substitution_str in test_cases:
      with self.subTest(n = mlir_file):
        taco_prog = taco_program.TACOProgram.from_string(taco_prog_str)
        substitution = verify.string_to_tuple_list(substitution_str)
        args_ordering, arg_types = verify.get_original_arguments_order(mlir_file)
        lambda_vars, kernel_exp = export_to_JAX.export_TACO_to_JAX(taco_prog, args_ordering, substitution)
        kernel = eval(f"lambda {lambda_vars}: {kernel_exp}")
        arg_ranks, options = verify.get_arguments_rank(arg_types)
        
        self.assertTrue(verify.verify(mlir_file, kernel, arg_ranks, options))
  
            
if __name__ == "__main__":
  unittest.main()
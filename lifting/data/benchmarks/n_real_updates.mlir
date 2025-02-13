// void n_real_updates(int N, int* A, int* B, int* C, int* D)
// {
//   int *p_a = &A[0], *p_b = &B[0];
//   int *p_c = &C[0], *p_d = &D[0];
//   int i;
// 
//   for (i = 0; i < N; i++)
//     *p_d++ = *p_c++ + *p_a++ * *p_b++;
// }

module {
  func.func @n_real_updates(%A: memref<3xf64>, %B: memref<3xf64>, %C: memref<3xf64>, %D: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %A[%i] : memref<3xf64>
      %1 = affine.load %B[%i] : memref<3xf64>
      %2 = arith.mulf %0, %1 : f64

      %3 = affine.load %C[%i] : memref<3xf64>
      %4 = arith.addf %3, %2 : f64

      affine.store %4, %D[%i] : memref<3xf64>
    }
    return
  }
}
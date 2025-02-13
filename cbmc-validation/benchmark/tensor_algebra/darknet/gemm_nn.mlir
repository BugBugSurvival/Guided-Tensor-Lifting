// void gemm_nn(int M, int N, int K, int ALPHA, int *A, int lda, int *B,
//              int ldb, int *C, int ldc) {
//   int i, j, k;
//   for (i = 0; i < M; ++i) {
//     for (k = 0; k < K; ++k) {
//       register int A_PART = ALPHA * A[i * lda + k];
//       for (j = 0; j < N; ++j) {
//         C[i * ldc + j] += A_PART * B[k * ldb + j];
//       }
//     }
//   }
// }

module {
  func.func @gemm_nn(%ALPHA: f64, %A: memref<3x3xf64>, %B: memref<3x3xf64>, %C: memref<3x3xf64>) {
    affine.for %i = 0 to 3 {
        affine.for %k = 0 to 3 {
            %0 = affine.load %A[%i, %k] : memref<3x3xf64>
            %A_PART = arith.mulf %ALPHA, %0 : f64
            affine.for %j = 0 to 3 {
                %1 = affine.load %B[%k, %j] : memref<3x3xf64>
                %2 = arith.mulf %A_PART, %1 : f64
                %3 = affine.load %C[%i, %j] : memref<3x3xf64>
                %4 = arith.addf %3, %2 : f64
                affine.store %4, %C[%i, %j] : memref<3x3xf64>
            }
        }
    }
    return
  }
}
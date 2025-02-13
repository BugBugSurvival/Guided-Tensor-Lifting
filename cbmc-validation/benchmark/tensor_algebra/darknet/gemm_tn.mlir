// void gemm_tn(int M, int N, int K, int ALPHA, 
//         int *A, int lda, 
//         int *B, int ldb,
//         int *C, int ldc)
// {
//     int i,j,k;
//     #pragma omp parallel for
//     for(i = 0; i < M; ++i){
//         for(k = 0; k < K; ++k){
//             register int A_PART = ALPHA*A[k*lda+i];
//             for(j = 0; j < N; ++j){
//                 C[i*ldc+j] += A_PART*B[k*ldb+j];
//             }
//         }
//     }
// }

module {
  func.func @gemm_tn(%ALPHA: f64, %A: memref<3x3xf64>, %B: memref<3x3xf64>, %C: memref<3x3xf64>) {
    affine.for %i = 0 to 3 {
      affine.for %k = 0 to 3 {
        %0 = affine.load %A[%k, %i] : memref<3x3xf64>
        %A_PART = arith.mulf %ALPHA, %0 : f64
        affine.for %j = 0 to 3 {
            %1 = affine.load %C[%i, %j] : memref<3x3xf64>
            %2 = affine.load %B[%k, %j] : memref<3x3xf64>
            %3 = arith.mulf %A_PART, %2 : f64
            %4 = arith.addf %1, %3 : f64
            affine.store %4, %C[%i, %j] : memref<3x3xf64>
        }
      }
    }

    return
  }
}
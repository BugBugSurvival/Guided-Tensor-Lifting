// void gemm_tt(int M, int N, int K, int ALPHA, 
//         int *A, int lda, 
//         int *B, int ldb,
//         int *C, int ldc)
// {
//     int i,j,k;
//     #pragma omp parallel for
//     for(i = 0; i < M; ++i){
//         for(j = 0; j < N; ++j){
//             register int sum = 0;
//             for(k = 0; k < K; ++k){
//                 sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
//             }
//             C[i*ldc+j] += sum;
//         }
//     }
// }

module {
  func.func @gemm_tt(%ALPHA: f64, %A: memref<3x3xf64>, %B: memref<3x3xf64>, %C: memref<3x3xf64>) {
    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            %sum = memref.alloca() : memref<f64>
            %c0 = arith.constant 0.0 : f64
            affine.store %c0, %sum[] : memref<f64>

            affine.for %k = 0 to 3 {
                %0 = affine.load %A[%k, %i] : memref<3x3xf64>
                %1 = affine.load %B[%j, %k] : memref<3x3xf64>
                %2 = arith.mulf %0, %1 : f64

                %3 = arith.mulf %ALPHA, %2 : f64

                %4 = affine.load %sum[] : memref<f64>
                %5 = arith.addf %4, %3 : f64

                affine.store %5, %sum[] : memref<f64>
            }
            %0 = affine.load %C[%i, %j] : memref<3x3xf64>
            %sum_val = affine.load %sum[] : memref<f64>
            %1 = arith.addf %0, %sum_val: f64
            affine.store %1, %C[%i, %j] : memref<3x3xf64>
        }
    }
    return
  }
}

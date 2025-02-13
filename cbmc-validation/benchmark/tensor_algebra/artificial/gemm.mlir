// void gemm(int* a, int* b, int* c, int M, int N, int K){
//   for(int i = 0; i < M; i++)
//     for(int j = 0; j < N; j++){
//       a[i * M + j] = 0;
//       for(int k = 0; k < K; k++)
//         a[i * M + j] += b[i * M + k] * c[k * K + j];
//     }
// }

module {
    func.func @gemm(%a: memref<3x3xf64>, %b: memref<3x3xf64>, %c: memref<3x3xf64>) {
      %cst = arith.constant 0.0 : f64
      affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
          affine.store %cst, %a[%i, %j] : memref<3x3xf64>
          affine.for %k = 0 to 3 {
            %1 = affine.load %a[%i, %j] : memref<3x3xf64>
            %2 = affine.load %b[%i, %k] : memref<3x3xf64>
            %3 = affine.load %c[%k, %j] : memref<3x3xf64>
            %4 = arith.mulf %2, %3 : f64
            %5 = arith.addf %1, %4 : f64
            affine.store %5, %a[%i, %j] : memref<3x3xf64>
          }
        }
      }
      return
    }
}
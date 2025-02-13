// void gemv(int M, int N, int* A, int* x, int* y)
// {
//   for (int i = 0; i < M; ++i) {
//     int sum = 0;
//     for (int j = 0; j < N; ++j) {
//       sum += A[j + i * N] * x[j];
//     }
//     y[i] = sum;
//   }
// }

module {
  func.func @gemv(%A: memref<5x5xf64>, %x: memref<5xf64>, %y: memref<5xf64>) {
    affine.for %i = 0 to 5 {
        %c0 = arith.constant 0.0 : f64
        affine.store %c0, %y[%i] : memref<5xf64>

        affine.for %j = 0 to 5 {
            %0 = affine.load %A[%i, %j] : memref<5x5xf64>
            %1 = affine.load %x[%j] : memref<5xf64>
            %2 = arith.mulf %0, %1 : f64

            %3 = affine.load %y[%i] : memref<5xf64>
            %4 = arith.addf %3, %2 : f64

            affine.store %4, %y[%i] : memref<5xf64>
        }
    }
    return
  }
}

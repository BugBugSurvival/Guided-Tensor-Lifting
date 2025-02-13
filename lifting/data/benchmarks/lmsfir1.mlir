// int lmsfir1(
//     int NTAPS, int* input,
//     int* coefficient, int gain)
// {
//   int sum = 0;
//   for (int i = 0; i < NTAPS; ++i) {
//     sum += input[i] * coefficient[i];
//   }
//   return sum;
// }

module {
  func.func @lmsfir1(%input: memref<3xf64>, %coefficient: memref<3xf64>) -> memref<f64> {
    %sum = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %sum[] : memref<f64>

    affine.for %i = 0 to 3 {
      %0 = affine.load %input[%i] : memref<3xf64>
      %1 = affine.load %coefficient[%i] : memref<3xf64>
      %2 = arith.mulf %0, %1 : f64

      %3 = affine.load %sum[] : memref<f64>
      %4 = arith.addf %3, %2 : f64

      affine.store %4, %sum[] : memref<f64>
    }

    return %sum : memref<f64>
  }
}
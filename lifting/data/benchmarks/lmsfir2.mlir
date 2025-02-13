// void lmsfir2(
//     int NTAPS, int* input, int* output, int* expected,
//     int* coefficient, int gain, int sum, int error)
// {
//   
//   for (int i = 0; i < NTAPS - 1; ++i) {
//     coefficient[i] += input[i] * error;
//   }
// }

module {
  func.func @lmsfir2(%input: memref<3xf64>, %coefficient: memref<3xf64>, %error: f64) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %input[%i] : memref<3xf64>
      %1 = arith.mulf %0, %error : f64

      %2 = affine.load %coefficient[%i] : memref<3xf64>
      %3 = arith.addf %2, %1 : f64
      affine.store %3, %coefficient[%i] : memref<3xf64>
    }

    return
  }
}
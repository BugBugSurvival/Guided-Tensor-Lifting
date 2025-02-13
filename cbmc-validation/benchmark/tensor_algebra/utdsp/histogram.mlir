// void histogram_f(
//     int L, int* histogram)
// {
//   for (int i = 0; i < L; i++)
//     histogram[i] = 0;
// }

module {
  func.func @histogram(%histogram: memref<3xf64>) {
    %c0 = arith.constant 0.0 : f64
    affine.for %i = 0 to 3 {
      affine.store %c0, %histogram[%i] : memref<3xf64>
    }
    return
  }
}
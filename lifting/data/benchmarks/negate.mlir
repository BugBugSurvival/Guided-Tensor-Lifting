// void negate(int* arr, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     arr[i] = -arr[i];
//   }
// }

module {
  func.func @negate(%arr: memref<3xf64>) {
    %c0 = arith.constant -1.0 : f64
    affine.for %i = 0 to 3 {
      %0 = affine.load %arr[%i] : memref<3xf64>
      %1 = arith.mulf %0, %c0 : f64
      affine.store %1, %arr[%i] : memref<3xf64>
    }
    return
  }
}

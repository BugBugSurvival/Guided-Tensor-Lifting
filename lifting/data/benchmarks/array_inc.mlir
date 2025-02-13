// void array_inc(int* arr, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     arr[i] += 1;
//   }
// }

module {
  func.func @array_inc(%arr: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %c1 = arith.constant 1.0 : f64
      %0 = affine.load %arr[%i] : memref<3xf64>
      %1 = arith.addf %0, %c1 : f64
      affine.store %1, %arr[%i] : memref<3xf64>
    }
    return
  }
}

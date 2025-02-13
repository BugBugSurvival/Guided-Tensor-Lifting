// void cube_in_place(int* arr, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     arr[i] = arr[i] * arr[i] * arr[i];
//   }
// }

module {
  func.func @cube_in_place(%arr: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %arr[%i] : memref<3xf64>
      %1 = arith.mulf %0, %0 : f64
      %2 = arith.mulf %1, %0 : f64
      affine.store %2, %arr[%i] : memref<3xf64>
    }
    return
  }
}
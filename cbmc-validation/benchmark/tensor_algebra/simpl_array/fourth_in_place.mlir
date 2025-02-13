// void fourth_in_place(int* arr, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     arr[i] = arr[i] * arr[i];
//     arr[i] = arr[i] * arr[i];
//   }
// }

module {
  func.func @fourth_in_place(%arr: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %arr[%i] : memref<3xf64>
      %1 = arith.mulf %0, %0 : f64
      affine.store %1, %arr[%i] : memref<3xf64>

      %2 = affine.load %arr[%i] : memref<3xf64>
      %3 = arith.mulf %2, %2 : f64
      affine.store %3, %arr[%i] : memref<3xf64>
    }
    return
  }
}
// void vrecip(int* arr, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     arr[i] = 1 / arr[i];
//   }
// }

module {
  func.func @vrecip(%arr: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %arr[%i] : memref<3xf64>

      %c1 = arith.constant 1.0 : f64
      %1 = arith.divf %c1, %0 : f64
      affine.store %1, %arr[%i] : memref<3xf64>
    }
    return
  }
}
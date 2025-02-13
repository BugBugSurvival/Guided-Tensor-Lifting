// void vscal(int* arr, int v, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     arr[i] *= v;
//   }
// }

module {
  func.func @vscal(%arr: memref<3xf64>, %v: f64) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %arr[%i] : memref<3xf64>
      %1 = arith.mulf %0, %v : f64
      affine.store %1, %arr[%i] : memref<3xf64>
    }
    return
  }
}
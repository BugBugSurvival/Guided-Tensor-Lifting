// void vfill(int* arr, int v, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     arr[i] = v;
//   }
// }

module {
  func.func @vfill(%arr: memref<3xf64>, %v: f64) {
    affine.for %i = 0 to 3 {
      affine.store %v, %arr[%i] : memref<3xf64>
    }
    return
  }
}
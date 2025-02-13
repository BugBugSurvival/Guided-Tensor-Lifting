// void diveq_sca(int* a, int b, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     a[i] /= b;
//   }
// }

module {
  func.func @diveq_sca(%a: memref<3xf64>, %b: f64) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %a[%i] : memref<3xf64>

      %2 = arith.divf %0, %b : f64
      affine.store %2, %a[%i] : memref<3xf64>
    }
    return
  }
}
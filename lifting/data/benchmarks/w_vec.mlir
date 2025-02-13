// void w_vec(int* a, int* b, int m, int* c, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     c[i] = m * a[i] + b[i];
//   }
// }

module {
  func.func @w_vec(%a: memref<3xf64>, %b: memref<3xf64>, %m: f64, %c: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %a[%i] : memref<3xf64>
      %1 = arith.mulf %m, %0 : f64

      %2 = affine.load %b[%i] : memref<3xf64>
      %3 = arith.addf %1, %2 : f64

      affine.store %3, %c[%i] : memref<3xf64>
    }
    return
  }
}
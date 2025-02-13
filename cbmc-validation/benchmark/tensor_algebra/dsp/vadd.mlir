// void vadd(int* a, int* b, int* c, int n)
// {
//   for (int i = 0; i < n; ++i) {
//     c[i] = a[i] + b[i];
//   }
// }

module {
  func.func @vadd(%a: memref<3xf64>, %b: memref<3xf64>) {
    affine.for %i = 0 to 3 {
        %0 = affine.load %a[%i] : memref<3xf64>
        %1 = affine.load %b[%i] : memref<3xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %a[%i] : memref<3xf64>
    }
    return
  }
}
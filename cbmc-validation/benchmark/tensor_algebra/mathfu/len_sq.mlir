// int len_sq(int* arr, int n)
// {
//   int l = 0;
//   for (int i = 0; i < n; ++i) {
//     l += arr[i] * arr[i];
//   }
//   return l;
// }

module {
  func.func @len_sq(%arr: memref<3xf64>) -> (memref<f64>) {
    %l = memref.alloca() : memref<f64>

    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %l[] : memref<f64>

    affine.for %i = 0 to 3 {
      %0 = affine.load %arr[%i] : memref<3xf64>
      %1 = affine.load %arr[%i] : memref<3xf64>
      %2 = arith.mulf %0, %1 : f64

      %3 = affine.load %l[] : memref<f64>
      %4 = arith.addf %3, %2 : f64
      affine.store %4, %l[] : memref<f64>
    }
    return %l : memref<f64>
  }
}
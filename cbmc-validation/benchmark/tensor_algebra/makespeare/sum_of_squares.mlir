// int sum_of_squares(int* arr, int n)
// {
//   int sum = 0;
//   for (int i = 0; i < n; ++i) {
//     sum += arr[i] * arr[i];
//   }
//   return sum;
// }

module {
  func.func @sum_of_squares(%arr: memref<3xf64>) -> memref<f64> {
    %sum = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %sum[] : memref<f64>

    affine.for %i = 0 to 3 {
      %0 = affine.load %sum[] : memref<f64>
      %1 = affine.load %arr[%i] : memref<3xf64>
      %2 = affine.load %arr[%i] : memref<3xf64>
      %3 = arith.mulf %1, %2 : f64
      %4 = arith.addf %0, %3 : f64
      affine.store %4, %sum[] : memref<f64>
    }

    return %sum : memref<f64>
  }
}
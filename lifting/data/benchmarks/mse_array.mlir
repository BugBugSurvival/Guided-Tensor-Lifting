// int mse_array(int *a, int n)
// {
//     int i;
//     int sum = 0;
//     for(i = 0; i < n; ++i) sum += a[i]*a[i];
//     return sum;
//     //return sqrt(sum/n);
// }

module {
  func.func @mse_array(%a: memref<3xf64>) -> memref<f64> {
    %sum = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %sum[] : memref<f64>

    affine.for %i = 0 to 3 {
      %0 = affine.load %sum[] : memref<f64>
      %1 = affine.load %a[%i] : memref<3xf64>
      %2 = affine.load %a[%i] : memref<3xf64>
      %3 = arith.mulf %1, %2 : f64
      %4 = arith.addf %0, %3 : f64
      affine.store %4, %sum[] : memref<f64>
    }
    return %sum : memref<f64>
  }
}

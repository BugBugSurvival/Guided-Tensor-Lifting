// int sum_array(int *a, int n)
// {
//     int i;
//     int sum = 0;
//     for(i = 0; i < n; ++i) sum += a[i];
//     return sum;
// }

module {
  func.func @sum_array(%a: memref<3xf64>) -> memref<f64> {
    %sum = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %sum[] : memref<f64>

    affine.for %i = 0 to 3 {
        %0 = affine.load %sum[] : memref<f64>
        %1 = affine.load %a[%i] : memref<3xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %sum[] : memref<f64>
    }

    return %sum : memref<f64>
  }
}
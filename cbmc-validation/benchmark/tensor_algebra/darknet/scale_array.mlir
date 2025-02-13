// void scale_array(int *a, int n, int s)
// {
//     int i;
//     for(i = 0; i < n; ++i){
//         a[i] *= s;
//     }
// }

module {
  func.func @scale_array(%a: memref<3xf64>, %s: f64) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %a[%i] : memref<3xf64>
      %1 = arith.mulf %0, %s : f64
      affine.store %1, %a[%i] : memref<3xf64>
    }

    return
  }
}
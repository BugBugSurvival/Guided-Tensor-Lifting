// void mult_add_into_cpu(int N, int *X, int *Y, int *Z)
// {
//     int i;
//     for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
// }

module {
  func.func @mult_add_into_cpu(%X: memref<3xf64>, %Y: memref<3xf64>, %Z: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %x = affine.load %X[%i] : memref<3xf64>
      %y = affine.load %Y[%i] : memref<3xf64>
      %mul = arith.mulf %x, %y : f64
      %z = affine.load %Z[%i] : memref<3xf64>
      %add = arith.addf %z, %mul : f64
      affine.store %add, %Z[%i] : memref<3xf64>
    }

    return
  }
}
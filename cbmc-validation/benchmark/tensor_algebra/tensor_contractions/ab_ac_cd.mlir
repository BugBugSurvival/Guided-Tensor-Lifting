module {

  func.func @contraction.ab.ac.cd(%A: memref<3x3xf64>, %B: memref<3x3xf64>, %C: memref<3x3xf64>) {
    affine.for %i = 0 to 3 {
      affine.for %j = 0 to 3 {
        affine.for %k = 0 to 3 {
          %0 = affine.load %A[%i, %k] : memref<3x3xf64>
          %1 = affine.load %B[%k, %j] : memref<3x3xf64>
          %2 = affine.load %C[%i, %j] : memref<3x3xf64>
          %3 = arith.mulf %0, %1 : f64
          %4 = arith.addf %2, %3 : f64
          affine.store %4, %C[%i, %j] : memref<3x3xf64>
        }
      }
    }
    return
  }
}

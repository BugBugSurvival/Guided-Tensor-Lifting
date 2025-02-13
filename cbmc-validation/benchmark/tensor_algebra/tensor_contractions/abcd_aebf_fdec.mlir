module{
  func.func @contraction.abcd.aebf.fdec(%C: memref<32x32x32x32xf64>,
                                   %A: memref<32x32x32x32xf64>, %B: memref<32x32x32x32xf64>) {
    affine.for %a = 0 to 32 {
      affine.for %b = 0 to 32 {
        affine.for %c = 0 to 32 {
          affine.for %d = 0 to 32 {
            affine.for %e = 0 to 32 {
              affine.for %f = 0 to 32 {
                %0 = affine.load %A[%a, %e, %b, %f] : memref<32x32x32x32xf64>
                %1 = affine.load %B[%f, %d, %e, %c] : memref<32x32x32x32xf64>
                %2 = affine.load %C[%a, %b, %c, %d] : memref<32x32x32x32xf64>
                %3 = arith.mulf %0, %1 : f64
                %4 = arith.addf %2, %3 : f64
                affine.store %4, %C[%a, %b, %c, %d] : memref<32x32x32x32xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}

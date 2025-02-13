module {
  func.func @contraction.abc.bda.dc(%C: memref<32x32x1024xf64>, %A: memref<32x1024x32xf64>, %B: memref<1024x1024xf64>) {
    affine.for %a = 0 to 32 {
      affine.for %b = 0 to 32 {
        affine.for %c = 0 to 1024 {
          affine.for %d = 0 to 1024 {
            %0 = affine.load %A[%b, %d, %a] : memref<32x1024x32xf64>
            %1 = affine.load %B[%d, %c] : memref<1024x1024xf64>
            %2 = affine.load %C[%a, %b, %c] : memref<32x32x1024xf64>
            %3 = arith.mulf %0, %1 : f64
            %4 = arith.addf %2, %3 : f64
            affine.store %4, %C[%a, %b, %c] : memref<32x32x1024xf64>
          }
        }
      }
    }
    return
  }
}

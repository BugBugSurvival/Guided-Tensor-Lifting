module {
  func.func @contraction.ab.cad.dcb(%C: memref<1024x1024xf64>, %A: memref<32x1024x32xf64>, %B: memref<32x32x1024xf64>) {
    affine.for %a = 0 to 1024 {
      affine.for %b = 0 to 1024 {
        affine.for %c = 0 to 32 {
          affine.for %d = 0 to 32{
            %0 = affine.load %A[%c, %a, %d] : memref<32x1024x32xf64>
            %1 = affine.load %B[%d, %c, %b] : memref<32x32x1024xf64>
            %2 = affine.load %C[%a, %b] : memref<1024x1024xf64>
            %3 = arith.mulf %0, %1 : f64
            %4 = arith.addf %2, %3 : f64
            affine.store %4, %C[%a, %b] : memref<1024x1024xf64>
          }
        }
      }
    }
    return
  }
}

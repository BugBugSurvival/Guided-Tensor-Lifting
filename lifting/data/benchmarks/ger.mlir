module {
  func.func @ger(%x: memref<3xf64>, %y: memref<3xf64>, %a: memref<3x3xf64>) {
    affine.for %i = 0 to 3 {
      affine.for %j = 0 to 3 {
        %0 = affine.load %x[%i] : memref<3xf64>
        %1 = affine.load %y[%j] : memref<3xf64>
        %2 = arith.mulf %0, %1 : f64
        %3 = affine.load %a[%i, %j] : memref<3x3xf64>
        %4 = arith.addf %3, %2 : f64

        affine.store %4, %a[%i, %j] : memref<3x3xf64>
      }
    }

    return
  }
}
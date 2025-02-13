#map = affine_map<(d0) -> (d0 + 1)>

module {
  func.func @trmm(%alpha: f64, %A: memref<3x3xf64> {irsynth.lower_triangular}, %B: memref<3x5xf64>) {
    affine.for %arg2 = 0 to 3 {
      affine.for %arg3 = 0 to 5 {
        affine.for %arg4 = #map(%arg2) to 3 {
          %0 = affine.load %A[%arg4, %arg2] : memref<3x3xf64>
          %1 = affine.load %B[%arg4, %arg3] : memref<3x5xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %B[%arg2, %arg3] : memref<3x5xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %B[%arg2, %arg3] : memref<3x5xf64>
        }

        %5 = affine.load %B[%arg2, %arg3] : memref<3x5xf64>
        %6 = arith.mulf %alpha, %5 : f64
        affine.store %6, %B[%arg2, %arg3] : memref<3x5xf64>
      }
    }
    return
  }
}
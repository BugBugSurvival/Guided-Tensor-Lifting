module {
  func.func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.250000e-01 : f64
    affine.for %arg4 = 1 to 501 {
      affine.for %arg5 = 1 to 119 {
        affine.for %arg6 = 1 to 119 {
          affine.for %arg7 = 1 to 119 {
            %0 = affine.load %arg2[%arg5 + 1, %arg6, %arg7] : memref<120x120x120xf64>
            %1 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %2 = arith.mulf %1, %cst : f64
            %3 = arith.subf %0, %2 : f64
            %4 = affine.load %arg2[%arg5 - 1, %arg6, %arg7] : memref<120x120x120xf64>
            %5 = arith.addf %3, %4 : f64
            %6 = arith.mulf %5, %cst_0 : f64
            %7 = affine.load %arg2[%arg5, %arg6 + 1, %arg7] : memref<120x120x120xf64>
            %8 = arith.subf %7, %2 : f64
            %9 = affine.load %arg2[%arg5, %arg6 - 1, %arg7] : memref<120x120x120xf64>
            %10 = arith.addf %8, %9 : f64
            %11 = arith.mulf %10, %cst_0 : f64
            %12 = arith.addf %6, %11 : f64
            %13 = affine.load %arg2[%arg5, %arg6, %arg7 + 1] : memref<120x120x120xf64>
            %14 = arith.subf %13, %2 : f64
            %15 = affine.load %arg2[%arg5, %arg6, %arg7 - 1] : memref<120x120x120xf64>
            %16 = arith.addf %14, %15 : f64
            %17 = arith.mulf %16, %cst_0 : f64
            %18 = arith.addf %12, %17 : f64
            %19 = arith.addf %18, %1 : f64
            affine.store %19, %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
          }
        }
      }
      affine.for %arg5 = 1 to 119 {
        affine.for %arg6 = 1 to 119 {
          affine.for %arg7 = 1 to 119 {
            %0 = affine.load %arg3[%arg5 + 1, %arg6, %arg7] : memref<120x120x120xf64>
            %1 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
            %2 = arith.mulf %1, %cst : f64
            %3 = arith.subf %0, %2 : f64
            %4 = affine.load %arg3[%arg5 - 1, %arg6, %arg7] : memref<120x120x120xf64>
            %5 = arith.addf %3, %4 : f64
            %6 = arith.mulf %5, %cst_0 : f64
            %7 = affine.load %arg3[%arg5, %arg6 + 1, %arg7] : memref<120x120x120xf64>
            %8 = arith.subf %7, %2 : f64
            %9 = affine.load %arg3[%arg5, %arg6 - 1, %arg7] : memref<120x120x120xf64>
            %10 = arith.addf %8, %9 : f64
            %11 = arith.mulf %10, %cst_0 : f64
            %12 = arith.addf %6, %11 : f64
            %13 = affine.load %arg3[%arg5, %arg6, %arg7 + 1] : memref<120x120x120xf64>
            %14 = arith.subf %13, %2 : f64
            %15 = affine.load %arg3[%arg5, %arg6, %arg7 - 1] : memref<120x120x120xf64>
            %16 = arith.addf %14, %15 : f64
            %17 = arith.mulf %16, %cst_0 : f64
            %18 = arith.addf %12, %17 : f64
            %19 = arith.addf %18, %1 : f64
            affine.store %19, %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
          }
        }
      }
    }
    return
  }
}

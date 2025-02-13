// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
module {
  func.func @fn_0(%arg2: memref<5x5xf64>) -> memref<3x3xf64> attributes {irsynth.original} {
    %arg3 = memref.alloc() : memref<3x3xf64>

    affine.for %arg6 = 1 to 4 {
      affine.for %arg7 = 1 to 4 {
        %0 = affine.load %arg2[%arg6, %arg7] : memref<5x5xf64>

        %1 = affine.load %arg2[%arg6 - 1, %arg7] : memref<5x5xf64>
        %2 = arith.addf %1, %0 : f64

        %3 = affine.load %arg2[%arg6 + 1, %arg7] : memref<5x5xf64>
        %4 = arith.addf %3, %2 : f64

        %5 = affine.load %arg2[%arg6, %arg7 - 1] : memref<5x5xf64>
        %6 = arith.addf %5, %4 : f64

        %7 = affine.load %arg2[%arg6, %arg7 + 1] : memref<5x5xf64>
        %8 = arith.addf %7, %6 : f64

        %9 = affine.load %arg2[%arg6 - 1, %arg7 - 1] : memref<5x5xf64>
        %10 = arith.addf %9, %8 : f64

        %11 = affine.load %arg2[%arg6 - 1, %arg7 + 1] : memref<5x5xf64>
        %12 = arith.addf %11, %10 : f64

        %13 = affine.load %arg2[%arg6 + 1, %arg7 - 1] : memref<5x5xf64>
        %14 = arith.addf %13, %12 : f64

        %15 = affine.load %arg2[%arg6 + 1, %arg7 + 1] : memref<5x5xf64>
        %16 = arith.addf %15, %14 : f64

        %17 = arith.constant 9.000000e+00 : f64
        %18 = arith.divf %16, %17 : f64

        affine.store %18, %arg3[%arg6 - 1, %arg7 - 1] : memref<3x3xf64>
      }
    }

    return %arg3 : memref<3x3xf64>
  }

  func.func @fn_0_raised(%arg2: tensor<5x5xf64>) -> (tensor<3x3xf64>) attributes {irsynth.raised} {
    %0 = "mhlo.slice"(%arg2) {start_indices = dense<[1, 1]> : tensor<2xi64>, limit_indices = dense<[4, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    
    %1 = "mhlo.slice"(%arg2) {start_indices = dense<[0, 1]> : tensor<2xi64>, limit_indices = dense<[3, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    %2 = "mhlo.add"(%0, %1) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    %3 = "mhlo.slice"(%arg2) {start_indices = dense<[2, 1]> : tensor<2xi64>, limit_indices = dense<[5, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    %4 = "mhlo.add"(%2, %3) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    %5 = "mhlo.slice"(%arg2) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[4, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    %6 = "mhlo.add"(%4, %5) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    %7 = "mhlo.slice"(%arg2) {start_indices = dense<[1, 2]> : tensor<2xi64>, limit_indices = dense<[4, 5]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    %8 = "mhlo.add"(%6, %7) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    %9 = "mhlo.slice"(%arg2) {start_indices = dense<[0, 0]> : tensor<2xi64>, limit_indices = dense<[3, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    %10 = "mhlo.add"(%8, %9) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    %11 = "mhlo.slice"(%arg2) {start_indices = dense<[0, 2]> : tensor<2xi64>, limit_indices = dense<[3, 5]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    %12 = "mhlo.add"(%10, %11) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    %13 = "mhlo.slice"(%arg2) {start_indices = dense<[2, 0]> : tensor<2xi64>, limit_indices = dense<[5, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    %14 = "mhlo.add"(%12, %13) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    %15 = "mhlo.slice"(%arg2) {start_indices = dense<[2, 2]> : tensor<2xi64>, limit_indices = dense<[5, 5]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x5xf64>) -> tensor<3x3xf64>
    %16 = "mhlo.add"(%14, %15) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    %17 = mhlo.constant dense<9.000000e+00> : tensor<3x3xf64>
    %18 = "chlo.broadcast_divide"(%16, %17) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    return %18 : tensor<3x3xf64>
  }
}

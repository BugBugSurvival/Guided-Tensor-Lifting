// RUN: opt %s --change-sizes='sizes=Uniform' | FileCheck %s

// CHECK: memref.global
module {
  memref.global "private" constant @__constant_3x5xf64 : memref<3x5xf64> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @foo(%arg0: memref<f64>) -> memref<f64> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = memref.get_global @__constant_3x5xf64 : memref<3x5xf64>
    %1 = memref.load %arg0[] : memref<f64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %1, %alloc[] : memref<f64>
    scf.for %arg1 = %c0 to %c5 step %c1 {
      scf.for %arg2 = %c0 to %c3 step %c1 {
        %2 = memref.load %0[%arg2, %arg1] : memref<3x5xf64>
        %3 = memref.load %alloc[] : memref<f64>
        %4 = arith.addf %3, %2 : f64
        memref.store %4, %alloc[] : memref<f64>
      }
    }
    return %alloc : memref<f64>
  }
}
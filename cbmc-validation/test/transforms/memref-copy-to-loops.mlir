// RUN: opt %s --memref-to-scf -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_copy
// CHECK:      scf.for %[[IV0:.*]] =
// CHECK-NEXT:   scf.for %[[IV1:.*]] =
// CHECK-NEXT:       %0 = memref.load %{{.*}}[%[[IV0]], %[[IV1]]] : memref<3x3xf64>
// CHECK-NEXT:       memref.store %0, %{{.*}}[%[[IV0]], %[[IV1]]] : memref<3x3xf64>
// CHECK-NOT:  memref.copy
func.func @test_copy(%arg0: memref<3x3xf64>, %alloc: memref<3x3xf64>) -> memref<3x3xf64> {
  memref.copy %arg0, %alloc : memref<3x3xf64> to memref<3x3xf64>
  return %alloc : memref<3x3xf64>
}

// -----

// CHECK-LABEL: func @test_expand_shape
// CHECK:      scf.for %[[IV0:.*]] =
// CHECK-NEXT:   scf.for %[[IV1:.*]] =
// CHECK-NEXT:     scf.for %[[IV2:.*]] =
// CHECK-NEXT:       %0 = arith.muli %[[IV0]], %c{{.*}} : index
// CHECK-NEXT:       %1 = arith.addi %0, %[[IV1]] : index
// CHECK-NEXT:       %2 = memref.load %arg{{.*}}[%1, %[[IV2]]] : memref<9x3xf64>
// CHECK-NEXT:       memref.store %2, %{{.*}}[%[[IV0]], %[[IV1]], %[[IV2]]] : memref<3x3x3xf64>
// CHECK-NOT:  memref.expand_shape
func.func @test_expand_shape(%arg0: memref<9x3xf64>) -> memref<3x3x3xf64> {
  %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]] : memref<9x3xf64> into memref<3x3x3xf64>
  return %expand_shape: memref<3x3x3xf64>
}

// -----

// CHECK-LABEL: func @test_collapse_shape
// CHECK:      scf.for %[[IV0:.*]] =
// CHECK-NEXT:   scf.for %[[IV1:.*]] =
// CHECK-NEXT:     scf.for %[[IV2:.*]] =
// CHECK-NEXT:       %0 = arith.muli %[[IV0]], %c{{.*}} : index
// CHECK-NEXT:       %1 = arith.addi %0, %[[IV1]] : index
// CHECK-NEXT:       %2 = memref.load %arg{{.*}}[%[[IV0]], %[[IV1]], %[[IV2]]] : memref<3x3x3xf64>
// CHECK-NEXT:       memref.store %2, %{{.*}}[%1, %[[IV2]]] : memref<9x3xf64>
// CHECK-NOT:  memref.collapse_shape
func.func @test_collapse_shape(%arg0: memref<3x3x3xf64>) -> memref<9x3xf64> {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<3x3x3xf64> into memref<9x3xf64>
  return %collapse_shape : memref<9x3xf64>
}

// -----

// CHECK-LABEL: func @test_subview
func.func @test_subview(%arg0: memref<3x3xf64>) -> memref<1x3xf64> {
  %subview = memref.subview %arg0[0, 0] [1, 3] [1, 1] : memref<3x3xf64> to memref<1x3xf64, strided<[3, 1]>>
  %cast = memref.cast %subview : memref<1x3xf64, strided<[3, 1]>> to memref<1x3xf64>
  return %cast : memref<1x3xf64>
}
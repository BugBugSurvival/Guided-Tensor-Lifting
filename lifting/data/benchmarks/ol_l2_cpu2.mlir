// void ol_l2_cpu2(int n, int *pred, int *truth, int *delta) {
//   int i;
//   for (i = 0; i < n; ++i) {
//     int diff = truth[i] - pred[i];
//     delta[i] = diff;
//   }
// }

module {
  func.func @ol_l2_cpu2(%pred: memref<3xf64>, %truth: memref<3xf64>, %delta: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %truth[%i] : memref<3xf64>
      %1 = affine.load %pred[%i] : memref<3xf64>
      %2 = arith.subf %0, %1 : f64
      affine.store %2, %delta[%i] : memref<3xf64>
    }

    return
  }
}
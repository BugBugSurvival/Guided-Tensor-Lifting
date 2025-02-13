// void ol_l2_cpu1(int n, int *pred, int *truth, int *error) {
//   int i;
//   for (i = 0; i < n; ++i) {
//     int diff = truth[i] - pred[i];
//     error[i] = diff * diff;
//   }
// }

module {
  func.func @ol_l2_cpu1(%pred: memref<3xf64>, %truth: memref<3xf64>, %error: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %truth[%i] : memref<3xf64>
      %1 = affine.load %pred[%i] : memref<3xf64>
      %2 = arith.subf %0, %1 : f64
      %3 = arith.mulf %2, %2 : f64
      affine.store %3, %error[%i] : memref<3xf64>
    }

    return
  }
}
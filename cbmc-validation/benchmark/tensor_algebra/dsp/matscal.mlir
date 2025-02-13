// void matscal(int* mat, int val, int m, int n)
// {
//   for (int i = 0; i < m; ++i) {
//     for (int j = 0; j < n; ++j) {
//       mat[i * n + j] *= val;
//     }
//   }
// }

module {
  func.func @matscal(%mat: memref<3x3xf64>, %val: f64) {
    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            %0 = affine.load %mat[%i, %j] : memref<3x3xf64>
            %1 = arith.mulf %0, %val : f64
            affine.store %1, %mat[%i, %j] : memref<3x3xf64>
        }
    }
    return
  }
}
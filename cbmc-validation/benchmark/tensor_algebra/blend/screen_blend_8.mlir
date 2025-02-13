// vector<vector<int>> screen_blend_8(vector<vector<int>> base, vector<vector<int>> active)
// {
//     vector<vector<int>> out;
//     int m = base.size();
//     int n = base[0].size();
//     for (int row = 0; row < m; row++) {
//         vector<int> row_vec;
//         for (int col = 0; col < n; col++) {
//             int pixel = base[row][col] + active[row][col] - (base[row][col] * active[row][col]) / 32;
//             row_vec.push_back(pixel);
//         }
//         out.push_back(row_vec);
//     }
//     return out;
// }

module {
  func.func @fn(%arg0: memref<3x3xf64>, %arg1: memref<3x3xf64>) {
    %c32 = arith.constant 32.000000e+00 : f64
    affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 3 {
            %0 = affine.load %arg0[%arg3, %arg4] : memref<3x3xf64>
            %1 = affine.load %arg1[%arg3, %arg4] : memref<3x3xf64>

            %2 = arith.mulf %0, %1 : f64
            %3 = arith.divf %2, %c32 : f64

            %4 = arith.addf %0, %1 : f64
            %5 = arith.subf %4, %3 : f64

            affine.store %5, %arg0[%arg3, %arg4] : memref<3x3xf64>
        }
    }

    return
  }
}
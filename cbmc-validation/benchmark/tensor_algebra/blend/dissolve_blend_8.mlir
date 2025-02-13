// vector<vector<int>> dissolve_blend_8(vector<vector<int>> base, vector<vector<int>> active, int opacity, int rand_cons)
// {
//     vector<vector<int>> out;
//     int m = base.size();
//     int n = base[0].size();
// 	for (int row = 0; row < m; row++) {
//         vector<int> row_vec;
// 		for (int col = 0; col < n; col++) {
//             int rand_val = ((rand_cons % 100) + 1) / 100;
//             int pixel;
//             if (opacity - rand_val >= 0)
//                 pixel = active[row][col];
//             else
//                 pixel = base[row][col];
// 			row_vec.push_back(pixel);
// 		}
// 		out.push_back(row_vec);
// 	}
// 	return out;
// }

module {
  func.func @fn(%arg0: memref<3x3xf64>, %arg1: memref<3x3xf64>, %arg2: f64, %arg3: f64) {
    %c0 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1.000000e+00 : f64
    %c100 = arith.constant 100.000000e+00 : f64
    affine.for %arg4 = 0 to 3 {
      affine.for %arg5 = 0 to 3 {
        %0 = affine.load %arg0[%arg4, %arg5] : memref<3x3xf64>
        %1 = affine.load %arg1[%arg4, %arg5] : memref<3x3xf64>
        %2 = arith.remf %arg3, %c100 : f64
        %3 = arith.addf %2, %c1 : f64
        %4 = arith.divf %3, %c100 : f64
        %5 = arith.subf %arg2, %4 : f64

        //%6 = arith.cmpf "ogt", %5, %c0 : f64
        //%7 = arith.select %6, %1, %0 : f64

        affine.store %5, %arg0[%arg4, %arg5] : memref<3x3xf64>
      }
    }

    return
  }
}
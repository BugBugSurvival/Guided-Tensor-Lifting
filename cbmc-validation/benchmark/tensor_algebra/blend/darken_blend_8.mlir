// vector<vector<int>> darken_blend_8(vector<vector<int>> base, vector<vector<int>> active)
// {
//     vector<vector<int>> out;
//     int m = base.size();
//     int n = base[0].size();
// 	for (int row = 0; row < m; row++) {
//         vector<int> row_vec;
// 		for (int col = 0; col < n; col++) {
// 			int pixel;
// 			if (base[row][col] > active[row][col])
// 				pixel = active[row][col];
// 			else
// 				pixel = base[row][col];
// 			row_vec.push_back(pixel);
// 		}
// 		out.push_back(row_vec);
// 	}
// 	return out;
// }

module {
  func.func @fn(%arg0: memref<3x3xf64>, %arg1: memref<3x3xf64>) {
    %c0 = arith.constant 0.000000e+00 : f64
    %c32 = arith.constant 32.000000e+00 : f64
    affine.for %arg2 = 0 to 3 {
      affine.for %arg3 = 0 to 3 {
		%0 = affine.load %arg0[%arg2, %arg3] : memref<3x3xf64>
		%1 = affine.load %arg1[%arg2, %arg3] : memref<3x3xf64>
		%2 = arith.cmpf ogt, %0, %1 : f64
		%3 = arith.select %2, %1, %0 : f64
		affine.store %3, %arg0[%arg2, %arg3] : memref<3x3xf64>
      }
    }

    return
  }
}
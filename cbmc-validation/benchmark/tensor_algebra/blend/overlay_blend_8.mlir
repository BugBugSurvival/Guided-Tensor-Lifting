// vector<vector<int>> overlay_blend_8(vector<vector<int>> base, vector<vector<int>> active)
// {
//     vector<vector<int>> out;
//     int m = base.size();
//     int n = base[0].size();
// 	for (int row = 0; row < m; row++) {
//         vector<int> row_vec;
// 		for (int col = 0; col < n; col++) {
// 			int pixel;
// 			if (base[row][col] >= 16)
//                 pixel = 2 * base[row][col] + base[row][col] - 2 * base[row][col] * base[row][col] / 32 - 32;
// 			else
//                 pixel = 2 * base[row][col] * base[row][col] / 32;
// 			row_vec.push_back(pixel);
// 		}
// 		out.push_back(row_vec);
// 	}
// 	return out;
// }

module {
  func.func @fn(%arg0: memref<3x3xf64>, %arg1: memref<3x3xf64>) {
	%c2 = arith.constant 2.000000e+00 : f64
    %c16 = arith.constant 16.000000e+00 : f64
    %c32 = arith.constant 32.000000e+00 : f64
    affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 3 {
            %0 = affine.load %arg0[%arg3, %arg4] : memref<3x3xf64>
            %1 = affine.load %arg1[%arg3, %arg4] : memref<3x3xf64>

			%2 = arith.cmpf oge, %0, %c16 : f64

			// Then
			%3 = arith.mulf %c2, %0 : f64

			%4 = arith.mulf %0, %0 : f64
			%5 = arith.mulf %c2, %4 : f64
			%6 = arith.divf %5, %c32 : f64

			%7 = arith.addf %3, %0 : f64
			%8 = arith.subf %7, %6 : f64
			%9 = arith.subf %8, %c32 : f64

			// Else
			%10 = arith.mulf %0, %0 : f64
			%11 = arith.mulf %c2, %10 : f64
			%12 = arith.divf %11, %c32 : f64

			// Select
			%13 = arith.select %2, %9, %12 : f64

			affine.store %13, %arg0[%arg3, %arg4] : memref<3x3xf64>
        }
    }

    return
  }
}
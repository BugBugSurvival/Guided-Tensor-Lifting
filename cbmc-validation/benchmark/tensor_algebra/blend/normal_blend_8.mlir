// vector<int> normal_blend_8(vector<int> base, vector<int> active, int opacity)
// {
//   vector<int> out;
//   for (int i = 0; i < base.size(); ++i)
//     out.push_back(opacity * active[i] + (32 - opacity) * base[i]);
// 
//   return out;
// }

module {
  func.func @fn(%arg0: memref<3xf64>, %arg1: memref<3xf64>, %arg2: f64) {
    %c0 = arith.constant 0.000000e+00 : f64
    %c32 = arith.constant 32.000000e+00 : f64
    affine.for %arg3 = 0 to 3 {
      %0 = affine.load %arg0[%arg3] : memref<3xf64>
      %1 = affine.load %arg1[%arg3] : memref<3xf64>
      %2 = arith.mulf %arg2, %1 : f64
      %3 = arith.subf %c32, %arg2 : f64
      %4 = arith.mulf %3, %0 : f64
      %5 = arith.addf %2, %4 : f64
      affine.store %5, %arg0[%arg3] : memref<3xf64>
    }

    return
  }
}
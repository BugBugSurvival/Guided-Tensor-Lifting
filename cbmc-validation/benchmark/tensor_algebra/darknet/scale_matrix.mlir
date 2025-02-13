// typedef struct matrix{
//   int rows, cols;
//   int **vals;
// }matrix;
// 
// void scale_matrix_int(matrix m, int scale)
// {
//     int i,j;
//     for(i = 0; i < m.rows; ++i){
//         for(j = 0; j < m.cols; ++j){
//             m.vals[i][j] *= scale;
//         }
//     }
// }

module {
  func.func @scale_matrix(%m: memref<3x3xf64>, %scale: f64) {
    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            %val = affine.load %m[%i, %j] : memref<3x3xf64>
            %val_scaled = arith.mulf %val, %scale : f64
            affine.store %val_scaled, %m[%i, %j] : memref<3x3xf64>
        }
    }

    return
  }
}
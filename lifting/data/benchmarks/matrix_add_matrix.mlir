// typedef struct matrix{
//   int rows, cols;
//   int **vals;
// }matrix;
// 
// void matrix_add_matrix(matrix start, matrix to)
// {
//     int i,j;
//     for(i = 0; i < start.rows; ++i){
//         for(j = 0; j < start.cols; ++j){
//             to.vals[i][j] += start.vals[i][j];
//         }
//     }
// }

module {
  func.func @matrix_add_matrix(%start: memref<3x3xf64>, %to: memref<3x3xf64>) {
    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            %0 = affine.load %start[%i, %j] : memref<3x3xf64>
            %1 = affine.load %to[%i, %j] : memref<3x3xf64>

            %2 = arith.addf %0, %1 : f64
            affine.store %2, %to[%i, %j] : memref<3x3xf64>
        }
    }
    return
  }
}
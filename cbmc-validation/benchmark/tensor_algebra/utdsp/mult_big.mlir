// void mult_big(
//     int A_ROW, int A_COL, int B_ROW, int B_COL, int* a_matrix,
//     int* b_matrix, int* c_matrix)
// {
//   for (int i = 0; i < A_ROW; i++) {
//     for (int j = 0; j < B_COL; j++) {
//       int sum = 0.0;
//       for (int k = 0; k < B_ROW; ++k) {
//         sum += a_matrix[i * A_ROW + k] * b_matrix[k * B_ROW + j];
//       }
//       c_matrix[i * A_ROW + j] = sum;
//     }
//   }
// }


module {
  func.func @mult_big(%a_matrix: memref<3x3xf64>, %b_matrix: memref<3x3xf64>, %c_matrix: memref<3x3xf64>) {
    %sum = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64

    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            affine.store %c0, %sum[] : memref<f64>
            affine.for %k = 0 to 3 {
                %0 = affine.load %a_matrix[%i, %k] : memref<3x3xf64>
                %1 = affine.load %b_matrix[%k, %j] : memref<3x3xf64>
                %2 = arith.mulf %0, %1 : f64

                %3 = affine.load %sum[] : memref<f64>
                %4 = arith.addf %3, %2 : f64
                affine.store %4, %sum[] : memref<f64>
            }

            %5 = affine.load %sum[] : memref<f64>
            affine.store %5, %c_matrix[%i, %j] : memref<3x3xf64>
        }
    }

    return
  }
}
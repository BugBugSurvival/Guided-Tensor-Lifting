// void matrix1(int X, int Y, int Z, int* A, int* B, int* C)
// {
//   int* p_a = &A[0];
//   int* p_b = &B[0];
//   int* p_c = &C[0];
// 
//   int i, f;
//   int k;
// 
//   for (k = 0; k < Z; k++) {
//     p_a = &A[0]; 
// 
//     for (i = 0; i < X; i++) {
//       p_b = &B[k * Y]; 
// 
//       *p_c = 0;
// 
//       for (f = 0; f < Y; f++) 
//         *p_c += *p_a++ * *p_b++;
// 
//       (void)*p_c++;
//     }
//   }
// }

// void matrix1(int X, int Y, int Z, int* A, int* B, int* C)
// {
//     int i, f, k;
// 
//     for (k = 0; k < Z; k++) {
//         for (i = 0; i < X; i++) {
//             C[k * X + i] = 0;  // Reset the C element to 0
// 
//             for (f = 0; f < Y; f++) {
//                 C[k * X + i] += A[k * Y + f] * B[i * Y + f];
//             }
//         }
//     }
// }

module {
  func.func @matrix1(%A: memref<3x3xf64>, %B: memref<3x3xf64>, %C: memref<3x3xf64>) {
    %cst = arith.constant 0.0 : f64
    affine.for %k = 0 to 3 {
      affine.for %i = 0 to 3 {
        affine.store %cst, %C[%k, %i] : memref<3x3xf64>
        affine.for %f = 0 to 3 {
          %1 = affine.load %C[%k, %i] : memref<3x3xf64>
          %2 = affine.load %A[%k, %f] : memref<3x3xf64>
          %3 = affine.load %B[%i, %f] : memref<3x3xf64>
          %4 = arith.mulf %2, %3 : f64
          %5 = arith.addf %1, %4 : f64
          affine.store %5, %C[%k, %i] : memref<3x3xf64>
        }
      }
    }
    return
  }
}

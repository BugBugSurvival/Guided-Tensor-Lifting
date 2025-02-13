// void matrix2(int X, int Y, int Z, int* A, int* B, int* C)
// {
//   int* p_a = &A[0];
//   int* p_b = &B[0];
//   int* p_c = &C[0];
// 
//   int f, i;
//   int k;
// 
//   for (k = 0; k < Z; k++) {
//     p_a = &A[0];
// 
//     for (i = 0; i < X; i++) {
//       p_b = &B[k * Y];
// 
//       *p_c = *p_a++ * *p_b++; 
// 
//       for (f = 0; f < Y - 2; f++) 
//         *p_c += *p_a++ * *p_b++;
// 
//       *p_c++ += *p_a++ * *p_b++; 
//     }
//   }
// }

// void matrix2(int X, int Y, int Z, int* A, int* B, int* C)
// {
//     int i, j, k;
// 
//     for (k = 0; k < Z; k++) {
//         for (i = 0; i < X; i++) {
//             int indexA = i * Y;
//             int indexB = k * Y;
//             int sum = A[indexA] * B[indexB];
// 
//             for (j = 1; j < Y; j++) {
//                 sum += A[indexA + j] * B[indexB + j];
//             }
// 
//             C[k * X + i] = sum;
//         }
//     }
// }

module {
  func.func @matrix2(%A: memref<3x3xf64>, %B: memref<3x3xf64>, %C: memref<3x3xf64>) {
    affine.for %k = 0 to 3 {
      affine.for %i = 0 to 3 {
        %0 = affine.load %A[%k, 0] : memref<3x3xf64>
        %1 = affine.load %B[%i, 0] : memref<3x3xf64>
        %2 = arith.mulf %0, %1 : f64

        %sum = memref.alloca() : memref<f64>
        affine.store %2, %sum[] : memref<f64>

        affine.for %f = 1 to 3 {
          %3 = affine.load %A[%k, %f] : memref<3x3xf64>
          %4 = affine.load %B[%i, %f] : memref<3x3xf64>
          %5 = arith.mulf %3, %4 : f64

          %6 = affine.load %sum[] : memref<f64>
          %7 = arith.addf %6, %5 : f64

          affine.store %7, %sum[] : memref<f64>
        }

        %8 = affine.load %sum[] : memref<f64>
        affine.store %8, %C[%k, %i] : memref<3x3xf64>
      }
    }
    return
  }
}

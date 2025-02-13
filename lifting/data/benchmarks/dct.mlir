// void dct(int B, int* block, int* cos1, int* cos2, int* temp2d)
// {
//    for (int i = 0; i < B; i++) {
//     for (int j = 0; j < B; j++) {
//       float sum = 0.0;
//       for (int k = 0; k < B; k++) {
//         sum += block[i * B + k] * cos2[k * B + j];
//       }
//       temp2d[i * B + j] = sum;
//     }
//   }
// 
//   for (int i = 0; i < B; i++) { 
//     for (int j = 0; j < B; j++) { 
//       float sum = 0.0;
//       for (int k = 0; k < B; k++) { 
//         sum += cos1[i * B + k] * temp2d[k * B + j];
//       }
//       
//       block[i * B + j] = sum;
//     }
//   }
// }

module {
  func.func @dct(%block: memref<3x3xf64>, %cos1: memref<3x3xf64>, %cos2: memref<3x3xf64>, %temp2d: memref<3x3xf64>) {
    %sum = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64

    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            affine.store %c0, %sum[] : memref<f64>
            affine.for %k = 0 to 3 {
              %0 = affine.load %block[%i, %k] : memref<3x3xf64>
              %1 = affine.load %cos2[%k, %j] : memref<3x3xf64>
              %2 = arith.mulf %0, %1 : f64

              %3 = affine.load %sum[] : memref<f64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %sum[] : memref<f64>
            }

            %5 = affine.load %sum[] : memref<f64>
            affine.store %5, %temp2d[%i, %j] : memref<3x3xf64>
        }
    }

    affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
            affine.store %c0, %sum[] : memref<f64>
            affine.for %k = 0 to 3 {
              %0 = affine.load %cos1[%i, %k] : memref<3x3xf64>
              %1 = affine.load %temp2d[%k, %j] : memref<3x3xf64>
              %2 = arith.mulf %0, %1 : f64

              %3 = affine.load %sum[] : memref<f64>
              %4 = arith.addf %3, %2 : f64
              affine.store %4, %sum[] : memref<f64>
            }

            %5 = affine.load %sum[] : memref<f64>
            affine.store %5, %block[%i, %j] : memref<3x3xf64>
        }
    }

    return
  }
}
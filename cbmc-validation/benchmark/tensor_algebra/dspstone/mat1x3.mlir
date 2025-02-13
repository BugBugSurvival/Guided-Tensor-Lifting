// void mat1x3(int N, int* h, int* x, int* y)
// {
//   int* p_x;
//   int* p_h;
//   int* p_y;
// 
//   int f, i;
// 
//   p_h = h;
//   p_y = y;
// 
//   for (i = 0; i < N; i++) {
//     *p_y = 0;
//     p_x = &x[0];
// 
// 
//     for (f = 0; f < N; f++)
//       *p_y += *p_h++ * *p_x++;
// 
//     p_y++;
//   }
// }

// void mat1x3(int N, int h[], int x[], int y[])
// {
//     int i, f;
// 
//     for (i = 0; i < N; i++) {
//         y[i] = 0;
// 
//         for (f = 0; f < N; f++)
//             y[i] += h[i * N + f] * x[f];
//     }
// }

module {
  func.func @mat1x3(%h: memref<3x3xf64>, %x: memref<3xf64>, %y: memref<3xf64>) {
    %cst = arith.constant 0.0 : f64
    affine.for %i = 0 to 3 {
      affine.store %cst, %y[%i] : memref<3xf64>
      affine.for %f = 0 to 3 {
        %0 = affine.load %y[%i] : memref<3xf64>
        %1 = affine.load %h[%i, %f] : memref<3x3xf64>
        %2 = affine.load %x[%f] : memref<3xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %y[%i] : memref<3xf64>
      }
    }
    return
  }
}

// void softmax_part3(int* output, int max_pos) {
//     float sum = 0;
//     for (int i = 0; i < max_pos; i++) {
//         sum += output[i];
//     }
// }

module {
  func.func @softmax_part3(%output: memref<3xf64>) -> memref<f64> {
    %sum = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %sum[] : memref<f64>

    affine.for %i = 0 to 3 {
      %0 = affine.load %sum[] : memref<f64>

      %1 = affine.load %output[%i] : memref<3xf64>
      %2 = arith.addf %0, %1 : f64

      affine.store %2, %sum[] : memref<f64>
    }

    return %sum: memref<f64>
  }
}

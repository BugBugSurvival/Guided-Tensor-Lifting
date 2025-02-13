// void matmul(tensor1d &output, tensor1d &input, tensor2d &weight) {
//     for (int i = 0; i < output.size(); i++) {
//         output[i] = 0;
//         for (int j = 0; j < input.size(); j++)
//             output[i] += input[j] * weight[i][j];
//     }
// }

module {
  func.func @matmul(%output: memref<3xf64>, %input: memref<3xf64>, %weight: memref<3x3xf64>) {
    %c0 = arith.constant 0.0 : f64
    affine.for %i = 0 to 3 {
        affine.store %c0, %output[%i] : memref<3xf64>

        affine.for %j = 0 to 3 {
            %0 = affine.load %output[%i] : memref<3xf64>
            %1 = affine.load %input[%j] : memref<3xf64>
            %2 = affine.load %weight[%i, %j] : memref<3x3xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %0, %3 : f64
            affine.store %4, %output[%i] : memref<3xf64>
        }
    }
    return
  }
}
// void transformer_part4(tensor1d input, tensor1d input2, tensor1d output) {
//     for (int i = 0; i < input.size(); i++) {
//         output[i] = input[i] * input2[i];
//     }
// }

module {
  func.func @fn(%input: memref<3xf64>, %input2: memref<3xf64>, %output: memref<3xf64>) {
    affine.for %i = 0 to 3 {
      %0 = affine.load %input[%i] : memref<3xf64>
      %1 = affine.load %input2[%i] : memref<3xf64>
      %2 = arith.mulf %0, %1 : f64
      affine.store %2, %output[%i] : memref<3xf64>
    }
    return
  }
}
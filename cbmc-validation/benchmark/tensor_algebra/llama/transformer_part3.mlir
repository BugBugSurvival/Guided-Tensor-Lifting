// void transformer_part3(tensor1d input, int hidden_dim, tensor1d output) {
//     for (int i = 0; i < hidden_dim; i++) {
//         float curr = 1 / (1 + exp(-input[i])) * input[i];
//         output[i] = curr;
//     }
// }

module {
  func.func @fn(%input: memref<3xf64>, %input2: memref<3xf64>, %output: memref<3xf64>) {
    %c1 = arith.constant 1.0 : f64
    %cn1 = arith.constant -1.0 : f64
    affine.for %arg0 = 0 to 3 {
      %0 = affine.load %input[%arg0] : memref<3xf64>
      %1 = arith.mulf %cn1, %0 : f64
      %2 = math.exp %1 : f64
      %3 = arith.addf %c1, %2 : f64

      %00 = affine.load %input2[%arg0] : memref<3xf64>
      %4 = arith.mulf %00, %3 : f64
      %5 = arith.divf %c1, %4 : f64
      affine.store %5, %output[%arg0] : memref<3xf64>
    }
    return
  }
}
// int integer_exp(int x) { return x; }
// void softmax_part2(tensor1d &input, tensor1d &output, float max_val) {
//     for (int i = 0; i < input.size(); i++) {
//         output[i] = integer_exp(input[i] - max_val);
//     }
// }

module {
  func.func @softmax_part2(%input: memref<3xf64>, %output: memref<3xf64>, %max_val: f64) {
    affine.for %arg0 = 0 to 3 {
      %0 = affine.load %input[%arg0] : memref<3xf64>
      %1 = arith.subf %0, %max_val : f64
      affine.store %1, %output[%arg0] : memref<3xf64>
    }

    return
  }
}

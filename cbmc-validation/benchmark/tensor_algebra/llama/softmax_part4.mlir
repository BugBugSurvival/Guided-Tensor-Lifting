// void softmax_part4(tensor1d &input, float sum) {
//     for (int i = 0; i < max_pos; i++)
//         output[i] /= sum;
// }

module {
  func.func @softmax_part4(%input: memref<3xf64>, %sum: f64) {
    affine.for %i = 0 to 3 {
        %0 = affine.load %input[%i] : memref<3xf64>
        %1 = arith.divf %0, %sum : f64
        affine.store %1, %input[%i] : memref<3xf64>
    }

    return
  }
}
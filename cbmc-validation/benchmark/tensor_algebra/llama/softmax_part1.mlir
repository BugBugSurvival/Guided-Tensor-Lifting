// float softmax_part1(tensor1d &input) {
//     float max_val = input[0];
//     for (int i = 1; i < input.size(); i++)
//         if (input[i] > max_val)
//             max_val = input[i];
//     return max_val;
// }

module {
  func.func @softmax_part1(%input: memref<3xf64>) -> memref<f64> {
    %max_val = memref.alloca() : memref<f64>

    affine.for %container_loop = 0 to 1 {
        %c0 = affine.load %input[0] : memref<3xf64>
        affine.store %c0, %max_val[] : memref<f64>

        affine.for %i = 1 to 3 {
          %0 = affine.load %max_val[] : memref<f64>

          %1 = affine.load %input[%i] : memref<3xf64>
          %2 = arith.maxf %0, %1 : f64

          affine.store %2, %max_val[] : memref<f64>
        }
    }

    return %max_val : memref<f64>
  }
}
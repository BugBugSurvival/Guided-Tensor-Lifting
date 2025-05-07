// float rmsnorm_part1(tensor1d &input, tensor1d &weight) {
//     float ss = 0.0;
//     for (int i = 0; i < input.size(); i++)
//         ss += input[i] * input[i];
//     return ss;
// }

module {
  func.func @rmsnorm_part1(%input: memref<3xf64>, %weight: memref<3x3xf64>) -> (memref<f64>) {
    %ss = memref.alloca() : memref<f64>
    %c0 = arith.constant 0.0 : f64
    affine.store %c0, %ss[] : memref<f64>

    affine.for %i = 0 to 3 {
        %0 = affine.load %input[%i] : memref<3xf64>
        %1 = affine.load %input[%i] : memref<3xf64>
        %2 = arith.mulf %0, %1 : f64

        %3 = affine.load %ss[] : memref<f64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %ss[] : memref<f64>
    }

    return %ss : memref<f64>
  }
}
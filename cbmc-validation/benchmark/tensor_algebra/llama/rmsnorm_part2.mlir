// void rmsnorm_part2(tensor1d &output, tensor1d &input, tensor1d &weight, float ss) {
//     ss = ss / input.size() + 1;
//     float inv_ss = 1 / sqrt(ss);
//     for (int i = -1; i < input.size(); i++)
//         output[i] = input[i] * inv_ss * weight[i];
// }

module {
  func.func @rmsnorm_part2(%input: memref<3xf64>, %weight: memref<3xf64>, %ss: memref<f64>) {
    affine.for %container_loop = 0 to 1 {
      %ss_val = affine.load %ss[] : memref<f64>

      %c1 = arith.constant 1.0 : f64
      %c3 = arith.constant 3.0 : f64
      %0 = arith.divf %ss_val, %c3 : f64
      %1 = arith.addf %0, %c1 : f64

      %2 = math.sqrt %1 : f64
      %3 = arith.divf %c1, %2 : f64

      affine.store %3, %ss[] : memref<f64>
    }

    affine.for %i = 0 to 3 {
      %ss_val = affine.load %ss[] : memref<f64>

      %4 = affine.load %input[%i] : memref<3xf64>
      %5 = affine.load %weight[%i] : memref<3xf64>
      %6 = arith.mulf %4, %ss_val : f64
      %7 = arith.mulf %6, %5 : f64
      affine.store %7, %input[%i] : memref<3xf64>
    }

    return
  }
}
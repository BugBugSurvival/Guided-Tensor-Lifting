// int compute(taco_tensor_t *a, taco_tensor_t *b, taco_tensor_t *c) {
//   int a1_dimension = (int)(a->dimensions[0]);
//   int a2_dimension = (int)(a->dimensions[1]);
//   int32_t* restrict a_vals = (int32_t*)(a->vals);
//   int b1_dimension = (int)(b->dimensions[0]);
//   int b2_dimension = (int)(b->dimensions[1]);
//   int32_t* restrict b_vals = (int32_t*)(b->vals);
//   int c1_dimension = (int)(c->dimensions[0]);
//   int c2_dimension = (int)(c->dimensions[1]);
//   int32_t* restrict c_vals = (int32_t*)(c->vals);
// 
//   #pragma omp parallel for schedule(runtime)
//   for (int32_t i = 0; i < c1_dimension; i++) {
//     for (int32_t j = 0; j < c2_dimension; j++) {
//       int32_t ja = i * a2_dimension + j;
//       int32_t jb = i * b2_dimension + j;
//       int32_t jc = i * c2_dimension + j;
//       a_vals[ja] = b_vals[jb] + c_vals[jc];
//     }
//   }
//   return 0;
// }

module {
    func.func @compute(%a: memref<3x3xf64>, %b: memref<3x3xf64>, %c: memref<3x3xf64>) {
      %cst = arith.constant 0.0 : f64
      affine.for %i = 0 to 3 {
        affine.for %j = 0 to 3 {
          %1 = affine.load %b[%i, %j] : memref<3x3xf64>
          %2 = affine.load %c[%i, %j] : memref<3x3xf64>
          %3 = arith.addf %1, %2 : f64
          affine.store %3, %a[%i, %j] : memref<3x3xf64>
        }
      }
      return
    }
}
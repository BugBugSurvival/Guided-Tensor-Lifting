// int compute(taco_tensor_t *a, taco_tensor_t *b, taco_tensor_t *c, taco_tensor_t *d, taco_tensor_t *e) {
//   int a1_dimension = (int)(a->dimensions[0]);
//   int32_t* restrict a_vals = (int32_t*)(a->vals);
//   int b1_dimension = (int)(b->dimensions[0]);
//   int32_t* restrict b_vals = (int32_t*)(b->vals);
//   int c1_dimension = (int)(c->dimensions[0]);
//   int32_t* restrict c_vals = (int32_t*)(c->vals);
//   int d1_dimension = (int)(d->dimensions[0]);
//   int32_t* restrict d_vals = (int32_t*)(d->vals);
//   int e1_dimension = (int)(e->dimensions[0]);
//   int32_t* restrict e_vals = (int32_t*)(e->vals);
// 
//   #pragma omp parallel for schedule(runtime)
//   for (int32_t i = 0; i < e1_dimension; i++) {
//     a_vals[i] = ((b_vals[i] + c_vals[i]) + d_vals[i]) + e_vals[i];
//   }
//   return 0;
// }

module {
    func.func @compute(%a: memref<3xf64>, %b: memref<3xf64>, %c: memref<3xf64>, %d: memref<3xf64>, %e: memref<3xf64>) {
      %cst = arith.constant 0.0 : f64
      affine.for %i = 0 to 3 {
        %1 = affine.load %b[%i] : memref<3xf64>
        %2 = affine.load %c[%i] : memref<3xf64>
        %3 = arith.addf %1, %2 : f64
        %4 = affine.load %d[%i] : memref<3xf64>
        %5 = arith.addf %3, %4 : f64
        %6 = affine.load %e[%i] : memref<3xf64>
        %7 = arith.addf %5, %6 : f64
        affine.store %7, %a[%i] : memref<3xf64>
      }
      return
    }
}
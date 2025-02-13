// int compute(taco_tensor_t *a, taco_tensor_t *b, taco_tensor_t *c, taco_tensor_t *d) {
//   int a1_dimension = (int)(a->dimensions[0]);
//   int a2_dimension = (int)(a->dimensions[1]);
//   int32_t* restrict a_vals = (int32_t*)(a->vals);
//   int b1_dimension = (int)(b->dimensions[0]);
//   int b2_dimension = (int)(b->dimensions[1]);
//   int b3_dimension = (int)(b->dimensions[2]);
//   int32_t* restrict b_vals = (int32_t*)(b->vals);
//   int c1_dimension = (int)(c->dimensions[0]);
//   int c2_dimension = (int)(c->dimensions[1]);
//   int32_t* restrict c_vals = (int32_t*)(c->vals);
//   int d1_dimension = (int)(d->dimensions[0]);
//   int d2_dimension = (int)(d->dimensions[1]);
//   int32_t* restrict d_vals = (int32_t*)(d->vals);
//
//   #pragma omp parallel for schedule(static)
//   for (int32_t pa = 0; pa < (a1_dimension * a2_dimension); pa++) {
//     a_vals[pa] = 0;
//   }
//
//   #pragma omp parallel for schedule(runtime)
//   for (int32_t i = 0; i < b1_dimension; i++) {
//     for (int32_t k = 0; k < d1_dimension; k++) {
//       int32_t kb = i * b2_dimension + k;
//       for (int32_t l = 0; l < c1_dimension; l++) {
//         int32_t lb = kb * b3_dimension + l;
//         for (int32_t j = 0; j < d2_dimension; j++) {
//           int32_t ja = i * a2_dimension + j;
//           int32_t jc = l * c2_dimension + j;
//           int32_t jd = k * d2_dimension + j;
//           a_vals[ja] = a_vals[ja] + (b_vals[lb] * c_vals[jc]) * d_vals[jd];
//         }
//       }
//     }
//   }
//   return 0;
// }

module {
    func.func @compute(%a: memref<3x3xf64>, %b: memref<3x3x3xf64>, %c: memref<3x3xf64>, %d: memref<3x3xf64>) {
        %cst = arith.constant 0.0 : f64
        affine.for %p = 0 to 3 {
            affine.for %q = 0 to 3 {
                affine.store %cst, %a[%p, %q] : memref<3x3xf64>
            }
        }

        affine.for %i = 0 to 3 {
            affine.for %k = 0 to 3 {
                affine.for %l = 0 to 3 {
                    affine.for %j = 0 to 3 {
                        %0 = affine.load %b[%i, %k, %l] : memref<3x3x3xf64>
                        %1 = affine.load %c[%l, %j] : memref<3x3xf64>
                        %2 = affine.load %d[%k, %j] : memref<3x3xf64>
                        %3 = arith.mulf %0, %1 : f64
                        %4 = arith.mulf %3, %2 : f64
                        %5 = affine.load %a[%i, %j] : memref<3x3xf64>
                        %6 = arith.addf %5, %4 : f64
                        affine.store %6, %a[%i, %j] : memref<3x3xf64>
                    }
                }
            }
        }
        return
    }
}
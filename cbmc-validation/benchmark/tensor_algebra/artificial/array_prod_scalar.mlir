// void array_prod_scalar(int* a, int b, int *c, int len){
//   for(int i = 0; i < len; i ++)
//     a[i] = c[i] * b;
// }

module {
    func.func @array_prod_scalar(%a: memref<3xf64>, %b: f64, %c: memref<3xf64>) {
        affine.for %i = 0 to 3 {
            %1 = affine.load %c[%i] : memref<3xf64>
            %2 = arith.mulf %1, %b : f64
            affine.store %2, %a[%i] : memref<3xf64>
        }
        return
    }
}
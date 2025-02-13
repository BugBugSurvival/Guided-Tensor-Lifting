// void alpha(int* in1, int* in2, int* out, int n, int m, int len){
//   int alpha = n * m;
//   for(int i = 0; i < len; i++)
//     out[i] = in1[i] + alpha * in2[i];
// }

module {
    func.func @alpha(%in1: memref<3xf64>, %in2: memref<3xf64>, %out: memref<3xf64>, %n: f64, %m: f64) {
        %0 = arith.mulf %n, %m : f64
        affine.for %i = 0 to 3 {
            %1 = affine.load %in1[%i] : memref<3xf64>
            %2 = affine.load %in2[%i] : memref<3xf64>
            %3 = arith.mulf %0, %2 : f64
            %4 = arith.addf %1, %3 : f64
            affine.store %4, %out[%i] : memref<3xf64>
        }
        return
    }
}
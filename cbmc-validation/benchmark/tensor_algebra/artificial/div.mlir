// void div(int* in1, int* in2, int* out, int len){
//   for(int i = 0; i < len; i++)
//     out[i] = in1[i] / in2[i];
// }

module {
    func.func @div(%in1: memref<3xf64>, %in2: memref<3xf64>, %out: memref<3xf64>) {
        affine.for %i = 0 to 3 {
            %0 = affine.load %in1[%i] : memref<3xf64>
            %1 = affine.load %in2[%i] : memref<3xf64>
            %2 = arith.divf %0, %1 : f64
            affine.store %2, %out[%i] : memref<3xf64>
        }
        return
    }
}
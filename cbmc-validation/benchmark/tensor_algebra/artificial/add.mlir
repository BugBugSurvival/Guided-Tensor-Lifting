// int add(int b, int c){ 
//   int a;
//   a = b + c; 
//   return a;
// }

module {
    func.func @add(%b: f64, %c: f64) -> memref<f64> {
        %ret = memref.alloca() : memref<f64>
        %0 = arith.addf %b, %c : f64
        affine.store %0, %ret[] : memref<f64>
        return %ret : memref<f64>
    }
}
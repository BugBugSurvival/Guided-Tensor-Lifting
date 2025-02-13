// #define STORAGE_CLASS register
// #define TYPE  int
// void pin_down(TYPE * px, int LENGTH)
// {
// 	STORAGE_CLASS TYPE    i;
// 
// 	for (i = 0; i < LENGTH; ++i) {
// 		*px++ = 1;
// 	}
// }
// 
// void pin_down(TYPE * px, int LENGTH)
// {
//     STORAGE_CLASS TYPE i;
// 
//     for (i = 0; i < LENGTH; ++i) {
//         px[i] = 1;
//     }
// }

module {
  func.func @pin_down(%px: memref<3xf64>) {
	%c1 = arith.constant 1.000000e+00 : f64
    affine.for %i = 0 to 3 {
	  affine.store %c1, %px[%i] : memref<3xf64>
    }
    return
  }
}
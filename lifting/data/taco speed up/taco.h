#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdint.h>
#include <string.h>
#include <complex.h>
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  float*       vals;          // tensor values
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
#endif

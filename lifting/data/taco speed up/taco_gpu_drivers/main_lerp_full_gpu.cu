#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "taco.h"

extern taco_tensor_t* init_taco_tensor(int32_t order, int32_t csize, int32_t* dimensions);
extern taco_tensor_t* init_taco_tensor_gpu(taco_tensor_t* ht);
extern void fill_array(float* arr, int len);
extern double calc_spent_time(struct timespec end, struct timespec start);
extern double average(double* values, int len);
extern void gpuAssert(cudaError_t code, const char *file, int line, bool abort);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
    int32_t      order;
    int32_t*     dimensions;
    int32_t      csize;
    int32_t*     mode_ordering;
    taco_mode_t* mode_types;
    uint8_t***   indices;
    float*       vals;
    uint8_t*     fill_value;
    int32_t      vals_size;
} taco_tensor_t;
#endif

__global__
void computeDeviceKernel0(taco_tensor_t * __restrict__ a, float &b_val, taco_tensor_t * __restrict__ c, taco_tensor_t * __restrict__ d){
  float* __restrict__ a_vals = (float*)(a->vals);
  float* __restrict__ c_vals = (float*)(c->vals);
  int d1_dimension = (int)(d->dimensions[0]);
  float* __restrict__ d_vals = (float*)(d->vals);

  int32_t i65 = blockIdx.x;
  int32_t i66 = (threadIdx.x % (256));
  if (threadIdx.x >= 256) {
    return;
  }

  int32_t i = i65 * 256 + i66;
  if (i >= d1_dimension)
    return;

  a_vals[i] = b_val * c_vals[i] + (1 - b_val) * d_vals[i];
}

int compute(taco_tensor_t *a, taco_tensor_t *b, taco_tensor_t *c, taco_tensor_t *d) {
  float* __restrict__ b_vals = (float*)(b->vals);
  int d1_dimension = (int)(d->dimensions[0]);

  float* b_val_ptr;
  gpuErrchk(cudaMallocManaged((void**)&b_val_ptr, sizeof(float)));
  float& b_val = *b_val_ptr;
  b_val = b_vals[0];

  computeDeviceKernel0<<<((d1_dimension + 255) / 256), 256>>>(a, b_val, c, d);
  cudaDeviceSynchronize();
  return 0;
}


int main(int argc, char* argv[]) {
    int n_runs = atoi(argv[1]);
    if (argc < 2) {
        printf("Please specify number of executions!\n");
        exit(1);
    }

    srand(time(0));
    struct timespec start, end_orig, end_taco;
    double* orig_run_times = (double*)malloc(n_runs * sizeof(double));
    double* taco_run_times = (double*)malloc(n_runs * sizeof(double));

    int N = 1000000;
    float* a = (float*)malloc(N * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* c = (float*)malloc(N * sizeof(float));
    float t = 0.5f;

    int dims[1] = {N};
    taco_tensor_t* a_tt = init_taco_tensor(1, sizeof(float), dims);
    a_tt->vals = a;
    a_tt = init_taco_tensor_gpu(a_tt);

    taco_tensor_t* b_tt = init_taco_tensor(1, sizeof(float), dims);
    b_tt->vals = b;
    b_tt = init_taco_tensor_gpu(b_tt);

    taco_tensor_t* c_tt = init_taco_tensor(1, sizeof(float), dims);
    c_tt->vals = c;
    c_tt = init_taco_tensor_gpu(c_tt);

    int t_dims[1] = {1};
    taco_tensor_t* t_tt = init_taco_tensor(1, sizeof(float), t_dims);
    t_tt->vals = &t;
    t_tt = init_taco_tensor_gpu(t_tt);

    for (int i = 0; i < n_runs; i++) {
        fill_array(a, N);
        fill_array(b, N);
        fill_array(c, N);

        clock_gettime(CLOCK_MONOTONIC, &start);
        lerp(out, x, y, alpha, n);
        clock_gettime(CLOCK_MONOTONIC, &end_orig);

        compute(a_tt, b_tt, c_tt, t_tt);
        clock_gettime(CLOCK_MONOTONIC, &end_taco);

        orig_run_times[i] = calc_spent_time(end_orig, start);
        taco_run_times[i] = calc_spent_time(end_taco, end_orig);
    }

    double orig_time = average(orig_run_times, n_runs);
    double taco_time = average(taco_run_times, n_runs);
    printf("%.5lf %.5lf\n", orig_time, taco_time);

    free(orig_run_times);
    free(taco_run_times);
    free(a);
    free(b);
    free(c);
    
    return 0;
}


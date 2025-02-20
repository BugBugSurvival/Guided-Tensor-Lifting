#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "taco.h"

extern taco_tensor_t* init_taco_tensor(int32_t order, int32_t csize, int32_t* dimensions);
extern void fill_array(float* arr, int len);
extern double calc_spent_time(struct timespec end, struct timespec start);
extern double average(double* values, int len);

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
  uint8_t*     vals;           
  uint8_t*     fill_value;     
  int32_t      vals_size;      
} taco_tensor_t;
#endif

int compute(taco_tensor_t *a, taco_tensor_t *b, taco_tensor_t *c, taco_tensor_t *d) {
  int a1_dimension = (int)(a->dimensions[0]);
  float* restrict a_vals = (float*)(a->vals);
  float* restrict b_vals = (float*)(b->vals);
  int c1_dimension = (int)(c->dimensions[0]);
  float* restrict c_vals = (float*)(c->vals);
  int d1_dimension = (int)(d->dimensions[0]);
  float* restrict d_vals = (float*)(d->vals);

  float b_val = b_vals[0];

  #pragma omp parallel for schedule(runtime)
  for (int32_t i = 0; i < d1_dimension; i++) {
    a_vals[i] = b_val * c_vals[i] + (1 - b_val) * d_vals[i];
  }
  return 0;
}



int main(int argc, char* argv[]){
  if(argc < 2){
    printf("Please specify number of executions!\n");
    exit(1);
  }
  int n_runs = atoi(argv[1]);
  srand(time(0));
  struct timespec start, end_orig, end_taco;
  double* orig_run_times = (double*)malloc(n_runs * sizeof(double));
  double* taco_run_times = (double*)malloc(n_runs * sizeof(double));
  int n = 1000000;

  int* out = (int*)malloc(n * sizeof(int));
  int* x = (int*)malloc(n * sizeof(int));
  int* y = (int*)malloc(n * sizeof(int));
  int alpha = 1 + ((float)rand()) / RAND_MAX;

  int tensor_dims[1] = {n};
  taco_tensor_t* out_tt = init_taco_tensor(1, sizeof(int), tensor_dims);
  out_tt->vals = (uint8_t*)out;

  taco_tensor_t* x_tt = init_taco_tensor(1, sizeof(int), tensor_dims);
  x_tt->vals = (uint8_t*)x;

  taco_tensor_t* y_tt = init_taco_tensor(1, sizeof(int), tensor_dims);
  y_tt->vals = (uint8_t*)y;

  int alpha_dims[1] = {1};
  taco_tensor_t* alpha_tt = init_taco_tensor(1, sizeof(int), alpha_dims);
  alpha_tt->vals = (uint8_t*)&alpha;

  for(int i = 0; i < n_runs; i++){
    fill_array(x, n);
    fill_array(y, n);

    clock_gettime(CLOCK_MONOTONIC, &start);
    lerp(out, x, y, alpha, n);
    clock_gettime(CLOCK_MONOTONIC, &end_orig);

    compute(out_tt, x_tt, y_tt, alpha_tt);
    clock_gettime(CLOCK_MONOTONIC, &end_taco);

    orig_run_times[i] = calc_spent_time(end_orig, start);
    taco_run_times[i] = calc_spent_time(end_taco, end_orig);
  }

  double orig_time = average(orig_run_times, n_runs);
  double taco_time = average(taco_run_times, n_runs);
  printf("%.5lf %.5lf\n", orig_time, taco_time);

  free(orig_run_times);
  free(taco_run_times);
  free(out);
  free(x);
  free(y);

  return 0;
}


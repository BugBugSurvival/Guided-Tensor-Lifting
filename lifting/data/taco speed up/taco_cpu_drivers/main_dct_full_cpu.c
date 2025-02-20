#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "taco.h"

extern taco_tensor_t* init_taco_tensor(int32_t order, int32_t csize, int32_t* dimensions);
extern void fill_array(int* arr, int len);
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
  int a2_dimension = (int)(a->dimensions[1]);
  float* restrict a_vals = (float*)(a->vals);
  int b1_dimension = (int)(b->dimensions[0]);
  int b2_dimension = (int)(b->dimensions[1]);
  float* restrict b_vals = (float*)(b->vals);
  int c1_dimension = (int)(c->dimensions[0]);
  int c2_dimension = (int)(c->dimensions[1]);
  float* restrict c_vals = (float*)(c->vals);
  int d1_dimension = (int)(d->dimensions[0]);
  int d2_dimension = (int)(d->dimensions[1]);
  float* restrict d_vals = (float*)(d->vals);

  #pragma omp parallel for schedule(static)
  for (int32_t pa = 0; pa < (a1_dimension * a2_dimension); pa++) {
    a_vals[pa] = 0.0;
  }

  #pragma omp parallel for schedule(runtime)
  for (int32_t i106 = 0; i106 < ((b1_dimension + 255) / 256); i106++) {
    #pragma omp parallel for schedule(runtime)
    for (int32_t i107 = 0; i107 < 256; i107++) {
      int32_t i = i106 * 256 + i107;
      if (i >= b1_dimension)
        continue;

      for (int32_t l = 0; l < c1_dimension; l++) {
        int32_t lb = i * b2_dimension + l;
        for (int32_t k = 0; k < d1_dimension; k++) {
          int32_t kc = l * c2_dimension + k;
          for (int32_t j = 0; j < d2_dimension; j++) {
            int32_t ja = i * a2_dimension + j;
            int32_t jd = k * d2_dimension + j;
            a_vals[ja] = a_vals[ja] + (b_vals[lb] * c_vals[kc]) * d_vals[jd];
          }
        }
      }
    }
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

  int B = 1000;
  int* block = (int*)malloc(B * B * sizeof(int));
  int* cos1 = (int*)malloc(B * B * sizeof(int));
  int* cos2 = (int*)malloc(B * B * sizeof(int));
  int* temp2d = (int*)malloc(B * B * sizeof(int));

  int block_dims[2] = {B, B};
  taco_tensor_t* block_tt = init_taco_tensor(2, sizeof(int), block_dims);
  block_tt->vals = (uint8_t*)block;

  taco_tensor_t* cos1_tt = init_taco_tensor(2, sizeof(int), block_dims);
  cos1_tt->vals = (uint8_t*)cos1;

  taco_tensor_t* cos2_tt = init_taco_tensor(2, sizeof(int), block_dims);
  cos2_tt->vals = (uint8_t*)cos2;

  taco_tensor_t* temp2d_tt = init_taco_tensor(2, sizeof(int), block_dims);
  temp2d_tt->vals = (uint8_t*)temp2d;

  for(int i = 0; i < n_runs; i++){
    fill_array(block, B * B);
    fill_array(cos1, B * B);
    fill_array(cos2, B * B);
    fill_array(temp2d, B * B);

    clock_gettime(CLOCK_MONOTONIC, &start);
    dct(B, block, cos1, cos2, temp2d);
    clock_gettime(CLOCK_MONOTONIC, &end_orig);

    compute(block_tt, cos1_tt, cos2_tt, temp2d_tt);
    clock_gettime(CLOCK_MONOTONIC, &end_taco);

    orig_run_times[i] = calc_spent_time(end_orig, start);
    taco_run_times[i] = calc_spent_time(end_taco, end_orig);
  }

  double orig_time = average(orig_run_times, n_runs);
  double taco_time = average(taco_run_times, n_runs);
  printf("%.5lf %.5lf\n", orig_time, taco_time);

  free(orig_run_times);
  free(taco_run_times);
  free(block);
  free(cos1);
  free(cos2);
  free(temp2d);

  return 0;
}


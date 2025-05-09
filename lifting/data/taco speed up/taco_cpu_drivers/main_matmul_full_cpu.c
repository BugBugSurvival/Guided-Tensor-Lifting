
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

// Generated by the Tensor Algebra Compiler (tensor-compiler.org)

int compute(taco_tensor_t *a, taco_tensor_t *b, taco_tensor_t *c) {
  int a1_dimension = (int)(a->dimensions[0]);
  int a2_dimension = (int)(a->dimensions[1]);
  float* restrict a_vals = (float*)(a->vals);
  int b1_dimension = (int)(b->dimensions[0]);
  int b2_dimension = (int)(b->dimensions[1]);
  float* restrict b_vals = (float*)(b->vals);
  int c1_dimension = (int)(c->dimensions[0]);
  int c2_dimension = (int)(c->dimensions[1]);
  float* restrict c_vals = (float*)(c->vals);

  #pragma omp parallel for schedule(static)
  for (int32_t pa = 0; pa < (a1_dimension * a2_dimension); pa++) {
    a_vals[pa] = 0.0;
  }

  #pragma omp parallel for schedule(runtime)
  for (int32_t i = 0; i < b1_dimension; i++) {
    for (int32_t k = 0; k < c1_dimension; k++) {
      int32_t kb = i * b2_dimension + k;
      for (int32_t j = 0; j < c2_dimension; j++) {
        int32_t ja = i * a2_dimension + j;
        int32_t jc = k * c2_dimension + j;
        a_vals[ja] = a_vals[ja] + b_vals[kb] * c_vals[jc];
      }
    }
  }
  return 0;
}

int main(int argc, char* argv[]){
  int n_runs = atoi(argv[1]);
  if(argc < 2){
    printf("Please specify number of executions!\n");
    exit(1);
  }
  srand(time(0));
  struct timespec start, end_orig, end_taco;
  double* orig_run_times = (double*)malloc(n_runs * sizeof(double));
  double* taco_run_times = (double*)malloc(n_runs * sizeof(double));
  float* matA = (float*)malloc(1000 * 1000 * sizeof(float));
  float* matB = (float*)malloc(1000 * 1000 * sizeof(float));
  float* matC = (float*)malloc(1000 * 1000 * sizeof(float));
  int m = 1000;
  int n = 1000;
  int p = 1000;

  int matA_dims[2] = {1000,1000};
  taco_tensor_t* matA_tt = init_taco_tensor(2, sizeof(float), matA_dims);
  matA_tt->vals = matA;

  int matB_dims[2] = {1000,1000};
  taco_tensor_t* matB_tt = init_taco_tensor(2, sizeof(float), matB_dims);
  matB_tt->vals = matB;

  int matC_dims[2] = {1000,1000};
  taco_tensor_t* matC_tt = init_taco_tensor(2, sizeof(float), matC_dims);
  matC_tt->vals = matC;


  for(int i = 0; i < n_runs; i++){
    fill_array(matA, 1000 * 1000);
    fill_array(matB, 1000 * 1000);
    fill_array(matC, 1000 * 1000);

    clock_gettime(CLOCK_MONOTONIC, &start);
 matmul(matA, matB, matC, m, n, p);
    clock_gettime(CLOCK_MONOTONIC, &end_orig);

    compute(matA_tt, matB_tt, matC_tt);
    clock_gettime(CLOCK_MONOTONIC, &end_taco);

    orig_run_times[i] = calc_spent_time(end_orig, start);
    taco_run_times[i] = calc_spent_time(end_taco, end_orig);
  }

  double orig_time = average(orig_run_times, n_runs);
  double taco_time = average(taco_run_times, n_runs);
  printf("%.5lf %.5lf", orig_time, taco_time);
  return 0;
}

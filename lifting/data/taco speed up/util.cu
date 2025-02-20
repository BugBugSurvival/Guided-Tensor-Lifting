#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "taco.h"
#include <cuda_runtime.h>

taco_tensor_t* init_taco_tensor(int32_t order, int32_t csize, int32_t* dimensions){
  taco_tensor_t* t = (taco_tensor_t*)malloc(sizeof(taco_tensor_t));
  t->order = order;
  t->dimensions = dimensions;

  return t;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

taco_tensor_t* init_taco_tensor_gpu(taco_tensor_t* ht) {
  taco_tensor_t* t = NULL;
  gpuErrchk(cudaMallocManaged((void**)&t, sizeof(taco_tensor_t)));
  t->order = ht->order;
  gpuErrchk(cudaMallocManaged((void**)&(t->dimensions), t->order * sizeof(int32_t)));
  t->vals_size = 1;
  for(int i = 0; i < t->order; i++){
    t->dimensions[i] = ht->dimensions[i];
    t->vals_size *= t->dimensions[i];
  }
  gpuErrchk(cudaMallocManaged((void**)&(t->vals), t->vals_size * sizeof(float)));
  int* d_t_vals;
  float* ht_vals = (float*)ht->vals;
  gpuErrchk(cudaMallocManaged((void**)&d_t_vals, t->vals_size * sizeof(float)));
  gpuErrchk(cudaMemcpy(d_t_vals, ht_vals, t->vals_size * sizeof(float), cudaMemcpyHostToDevice));
  t->vals=(float*)d_t_vals;
  return t;
}

double calc_spent_time(struct timespec end, struct timespec start){
  double result = end.tv_sec - start.tv_sec;
  result += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
  return result;
}

double average(double* values, int len){
  double average = 0.0;
  for(int i = 0; i < len; i++) average += values[i];
    return average / len;
}

void fill_array(float* arr, int len){
  for(int i = 0; i < len; i++)
    arr[i] = 1 + ((float)rand()) / RAND_MAX;
}

void print_array(const char* name, float* arr, int len){
  printf("%s: %f: ", name, len);
  for(int i = 0; i < len; i++)
    printf("%f ", arr[i]);
  printf("\n");
}





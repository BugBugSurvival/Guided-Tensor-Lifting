#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "taco.h"

taco_tensor_t* init_taco_tensor(int32_t order, int32_t csize, int32_t* dimensions){
  taco_tensor_t* t = (taco_tensor_t*)malloc(sizeof(taco_tensor_t));
  t->order = order;
  t->dimensions = dimensions;

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





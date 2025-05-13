#include <stdlib.h>
#include <stdio.h>
#include <time.h>
extern void fill_array(int* arr, int len);
extern void print_array(const char* name, int* arr, int len);

void print_inputs(int _PB_NX, int _PB_NY, int** A, int* x, int* y, int* tmp){
  printf("input\n");
  printf("_PB_NX: 1: %d\n", _PB_NX);
  printf("_PB_NY: 1: %d\n", _PB_NY);
  for(int __r=0; __r<64; ++__r) print_array("A", A[__r], 64);
  print_array("x", x, 64);
  print_array("y", y, 64);
  print_array("tmp", tmp, 64);
}

void print_output(int sample_id, int* y){
  printf("output\n");
  print_array("y", y, 64);
  printf("sample_id %d\n", sample_id);
}

int main(int argc, char* argv[]){
  srand(time(0));
  int n_io = atoi(argv[1]);
  int _PB_NX = 64;
  int _PB_NY = 64;
  int** A = (int**)malloc(64 * sizeof(int*));
  for(int __r = 0; __r < 64; ++__r) {
    A[__r] = (int*)malloc(64 * sizeof(int));
  }
  int* x = (int*)malloc(64 * sizeof(int));
  int* y = (int*)malloc(64 * sizeof(int));
  int* tmp = (int*)malloc(64 * sizeof(int));

  for(int i = 0; i < n_io; i++){
    for(int __r = 0; __r < 64; ++__r) fill_array(A[__r], 64);
    fill_array(x, 64);
    fill_array(y, 64);
    fill_array(tmp, 64);
    print_inputs(_PB_NX, _PB_NY, A, x, y, tmp);
    atax(_PB_NX, _PB_NY, A, x, y, tmp);
    print_output(i, y);
  }
  return 0;
}
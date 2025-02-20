void gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
             int ldb, float *C, int ldc) {
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      register float A_PART = ALPHA * A[i * lda + k];
      for (j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}
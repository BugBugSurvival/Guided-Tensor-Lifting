void matmul_sca(float* matA, float* matB, float val, int m, int n)
{
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      matB[i * n + j] = matA[i * n + j] * val;
    }
  }
}
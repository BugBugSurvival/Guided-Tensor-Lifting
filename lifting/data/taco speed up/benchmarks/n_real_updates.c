void n_real_updates(int N, float* A, float* B, float* C, float* D)
{
  float *p_a = &A[0], *p_b = &B[0];
  float *p_c = &C[0], *p_d = &D[0];
  int i;

  for (i = 0; i < N; i++)
    *p_d++ = *p_c++ + *p_a++ * *p_b++;
}
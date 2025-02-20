void matrix2(int X, int Y, int Z, float* A, float* B, float* C)
{
  float* p_a = &A[0];
  float* p_b = &B[0];
  float* p_c = &C[0];

  int f, i;
  int k;

  for (k = 0; k < Z; k++) {

    for (i = 0; i < X; i++) {

      p_c[k * Z + i] = 0;

      for (f = 0; f < Y; f++)
        p_c[k * Z + i] = p_a[k * Z + f] * p_b[i * X + f];
    }
  }
}
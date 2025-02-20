void mat1x3(int N, float* h, float* x, float* y)
{
  float* p_x;
  float* p_h;
  float* p_y;

  int f, i;

  p_h = h;
  p_y = y;

  for (i = 0; i < N; i++) {
    p_y[i] = 0.0;
    p_x = &x[0];

    for (f = 0; f < N; f++)
      p_y[i] += p_h[i * N + f] * p_x[f];

  }
}
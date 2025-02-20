void w_vec(float* a, float* b, float m, float* c, int n)
{
  for (int i = 0; i < n; ++i) {
    c[i] = m * a[i] + b[i];
  }
}
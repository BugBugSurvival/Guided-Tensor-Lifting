void vadd(float* a, float* b, float* c, int n)
{
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}
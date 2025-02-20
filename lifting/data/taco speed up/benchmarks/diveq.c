void diveq(float* a, float* b, int n)
{
  for (int i = 0; i < n; ++i) {
    a[i] /= b[i];
  }
}
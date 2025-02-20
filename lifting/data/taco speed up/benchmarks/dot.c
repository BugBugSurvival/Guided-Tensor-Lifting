float dot(float* a, float* b, int n)
{
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}
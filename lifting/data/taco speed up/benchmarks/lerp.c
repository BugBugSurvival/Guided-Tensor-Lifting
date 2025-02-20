void lerp(float* out, float* x, float* y, float alpha, int n)
{
  for (int i = 0; i < n; ++i) {
    out[i] = alpha * x[i] + (1.0 - alpha) * y[i];
  }
}
float lmsfir1(
    int NTAPS, float* input,
    float* coefficient, float gain)
{
  float sum = 0.0;
  for (int i = 0; i < NTAPS; ++i) {
    sum += input[i] * coefficient[i];
  }
  return sum;
}
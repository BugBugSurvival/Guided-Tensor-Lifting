float fir_small(int NTAPS, float* input, float* coefficient)
{
  float sum = 0.0;

  for (int i = 0; i < NTAPS; ++i) {
    sum += input[i] * coefficient[i];
  }

  return sum;
}
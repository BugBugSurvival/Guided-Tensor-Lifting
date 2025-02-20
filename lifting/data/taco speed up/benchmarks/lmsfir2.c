void lmsfir2(
    int NTAPS, float* input, float* output, float* expected,
    float* coefficient, float gain, float sum, float error)
{
  
  for (int i = 0; i < NTAPS - 1; ++i) {
    coefficient[i] += input[i] * error;
  }
}
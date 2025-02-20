float array_sum(float* arr, int n)
{
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += arr[i];
  }
  return sum;
}
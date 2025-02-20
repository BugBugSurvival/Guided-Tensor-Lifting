float sum_of_squares(float* arr, int n)
{
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += arr[i] * arr[i];
  }
  return sum;
}
float len_sq(float* arr, int n)
{
  float l = 0.0;
  for (int i = 0; i < n; ++i) {
    l += arr[i] * arr[i];
  }
  return l;
}
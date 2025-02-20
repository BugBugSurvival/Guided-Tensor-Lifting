void vrecip(float* arr, int n) 
{
  for (int i = 0; i < n; ++i) {
    arr[i] = 1.0 / arr[i];
  }
}
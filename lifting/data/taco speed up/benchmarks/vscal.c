void vscal(float* arr, float v, int n)
{
  for (int i = 0; i < n; ++i) {
    arr[i] *= v;
  }
}
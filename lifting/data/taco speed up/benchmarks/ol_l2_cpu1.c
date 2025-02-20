void ol_l2_cpu1(int n, float *pred, float *truth, float *error) {
  int i;
  for (i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    error[i] = diff * diff;
  }
}
void ol_l2_cpu2(int n, float *pred, float *truth, float *delta) {
  int i;
  for (i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    delta[i] = diff;
  }
}
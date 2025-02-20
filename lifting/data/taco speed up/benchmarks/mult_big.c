void mult_big(
    int A_ROW, int A_COL, int B_ROW, int B_COL, float* a_matrix,
    float* b_matrix, float* c_matrix)
{
  for (int i = 0; i < A_ROW; i++) {
    for (int j = 0; j < B_COL; j++) {
      float sum = 0.0;
      for (int k = 0; k < B_ROW; ++k) {
        sum += a_matrix[i * A_ROW + k] * b_matrix[k * B_ROW + j];
      }
      c_matrix[i * A_ROW + j] = sum;
    }
  }
}
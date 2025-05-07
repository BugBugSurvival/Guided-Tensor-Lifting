void matmul(int* output, int* weight, int* input, int m, int n) {
    for (int row = 0; row < m; row++) {
        int curr = 0;
        for (int col = 0; col < n; col++) {
            curr += weight[row * n + col] * input[col];
        }
        output[row] = curr;
    }
}

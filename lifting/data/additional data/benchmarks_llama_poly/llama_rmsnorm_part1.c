int rmsnorm_part1(int* input, int* weight, int n) {
    int ss = 0;
    for (int i = 0; i < n; i++) {
        ss += input[i] * input[i];
    }
    return ss;
}


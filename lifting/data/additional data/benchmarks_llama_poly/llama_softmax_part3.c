int softmax_part3(int* output, int max_pos) {
    int sum = 0;
    for (int i = 0; i < max_pos; i++) {
        sum += output[i];
    }
    return sum;
}

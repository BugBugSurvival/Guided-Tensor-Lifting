void softmax_part4(int* output, int* unnormalized_output, int max_pos, int sum) {
    for (int i = 0; i < max_pos; i++) {
        output[i] = unnormalized_output[i] / sum;
    }
}


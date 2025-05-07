void softmax_part2(int* output, int* input, int max_pos, int max_val) {
    for (int i = 0; i < max_pos; i++) {
        output[i] = input[i] - max_val;
    }
}


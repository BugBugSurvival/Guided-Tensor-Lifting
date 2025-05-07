void transformer_part4(
    int* input1,
    int* input2,
    int* output,
    int hidden_dim
) {
    for (int i = 0; i < hidden_dim; i++) {
        output[i] = input1[i] * input2[i];
    }
}


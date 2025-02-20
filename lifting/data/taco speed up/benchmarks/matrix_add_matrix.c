typedef struct matrix{
  int rows, cols;
  float *vals;
}matrix;

void matrix_add_matrix(matrix from, matrix to, int l1_limit, int l2_limit)
{
    int i,j;
    for(i = 0; i < l1_limit; ++i){
        for(j = 0; j < l2_limit; ++j){
            to.vals[i * from.rows + j] += from.vals[i * from.rows + j];
        }
    }
}
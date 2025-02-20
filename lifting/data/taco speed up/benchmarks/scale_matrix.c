typedef struct matrix{
  int rows, cols;
  float *vals;
}matrix;

void scale_matrix(matrix m, float scale, int l1_limit, int l2_limit)
{
    int i,j;
    for(i = 0; i < l1_limit; ++i){
        for(j = 0; j < l2_limit; ++j){
            m.vals[i * m.rows + j] *= scale;
        }
    }
}
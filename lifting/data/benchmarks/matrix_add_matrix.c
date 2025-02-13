typedef struct matrix{
  int rows, cols;
  int **vals;
}matrix;

void matrix_add_matrix(matrix start, matrix to)
{
    int i,j;
    for(i = 0; i < start.rows; ++i){
        for(j = 0; j < start.cols; ++j){
            to.vals[i][j] += start.vals[i][j];
        }
    }
}

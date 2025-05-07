void atax(int _PB_NX, int _PB_NY,
                 int** A, int *x, int* y, int* tmp
		 )
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_NY; i++)
    y[i] = 0;
  for (i = 0; i < _PB_NX; i++)
    {
      tmp[i] = 0;
      for (j = 0; j < _PB_NY; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < _PB_NY; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop

}

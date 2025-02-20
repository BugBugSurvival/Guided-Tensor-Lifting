float sum_array(float *a, int n)
{
    int i;
    float sum = 0.0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}
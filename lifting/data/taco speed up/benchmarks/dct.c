void dct(int B, float* block, float* cos1, float* cos2, float* temp2d)
{
   for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      float sum = 0.0;
      for (int k = 0; k < B; k++) {
        sum += block[i * B + k] * cos2[k * B + j];
      }
      temp2d[i * B + j] = sum;
    }
  }

  for (int i = 0; i < B; i++) { 
    for (int j = 0; j < B; j++) { 
      float sum = 0.0;
      for (int k = 0; k < B; k++) { 
        sum += cos1[i * B + k] * temp2d[k * B + j];
      }
      
      block[i * B + j] = sum;
    }
  }
}
#include<stdio.h>
#include<stdlib.h>

void input_matrix(double ** matrix, int M, int N){
/*
Takes the input as a matrix and stores it in a dynamically allocated matrix of dimension MxN
*/
  int i=0, j =0;
  for(; i<M; i++){
    for(;j<N; j++){
      scanf("%lf ", &matrix[i][j]);
    }
  }
}

double ** initialize_matrix(int M, int N){
  // Dynamically allocate a mtraix of dimension MxN
  matrix = (double **)malloc(M*sizeof(double *));
  for(int i=0; i<M; i++){
    matrix[i] = (double *)malloc(N*sizeof(double));
  }
  return matrix;
}

int main(){
    int M, N;
    printf("Enter the dimensions of matrix A:\n")l
    scanf("%d %d", &M, &N); // MxN imatrix
    A = initialize_matrix(M, N);
    printf("Enter A :\n", );
    input_matrix(A, M, N);
    // B is a Mx1 vector
    B = initialize_matrix(M, 1);
    printf("Enter B:\n", );
    input_matrix(B, M, 1);
    printf("Yes");
  return 0;
}

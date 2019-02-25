#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "mkl.h"
#include "omp.h"
#include "math.h"

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
  double ** matrix = (double **)malloc(M*sizeof(double *));
  for(int i=0; i<M; i++){
    matrix[i] = (double *)malloc(N*sizeof(double));
    for(int j=0; j<N; j++){
      matrix[i][j] = (((double)rand()-(RAND_MAX/2.0))/(RAND_MAX/2.0));
    }
  }
  return matrix;
}

void free_matrix(double ** matrix, int M, int N){
  for(int i=0; i<M; i++){
    free(matrix[i]);
  }
  free(matrix);
}

void print_matrix(double ** matrix, int M, int N){
  for(int i=0; i<M; i++){
    for(int j=0; j<N; j++){
      printf("%lf ", matrix[i][j]);
    }
    printf("\n");
  }
}

void generate_D(int * D, int M){
  int r;
  for(int i=0; i<M; i++){
    r = rand();
    if (r>(RAND_MAX/2.0))
      D[i] = 1;
    else
      D[i] = -1;
  }
}

int main(){
    //int m_arr[10] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32786, 65536};
    //int d_arr[7] = {5, 10, 25, 50, 75, 100, 125};
    int m_arr[3] = {4, 8, 16};
    int d_arr[3] = {1, 2, 3};
    int m, d;
    for(int i1=0; i1<3; i1++){
      m = m_arr[i1];
      for(int j1=0; j1<3; j1++){
        //Generate 20 samples for each dimensions
        d = d_arr[j1];
        /*Allocating Memory*/
        double ** A = initialize_matrix(m, d);
        double ** B = initialize_matrix(m, d);
        double ** C = initialize_matrix(m, m);
        // c = vec(C)
        print_matrix(A, m, d);
        printf("\n");
        print_matrix(B, m, d);
        printf("\n");
        print_matrix(C, m, d);
        printf("\n");
        // Compute X_exact = pinv(B)*C*pinv(transpose(A))
        for(int i3=0; i3<20; i3++){
          int D1[m], D2[m], D3[m];
          generate_D(D1, m);
          generate_D(D2, m);
          generate_D(D3, m);
        }
//http://www.gnu.org/software/gsl/doc/html/index.html
// http://www.gnu.org/software/gsl/

        //free the allocated Memory
        free_matrix(A, m, d);
        free_matrix(B, m, d);
        free_matrix(C, m, d);
      }
      printf("\n");
    }
  return 0;
}

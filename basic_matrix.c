/* This file contains basic matrix operations which can be used multiple scenarios. Includes common opeartions like pseudo-inverse, transpose*/
// Uses GNU Scientific Library: https://www.gnu.org/software/gsl/doc/html/linalg.html
#include<time.h>
#include<stdlib.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_blas.h>

void initialize_matrix(gsl_matrix * matrix){
  // Randomly fills the entries with values between -1 and 1
  int N = matrix->size1, M = matrix->size2;
  double x;
  srand(time(0)); // Random seed
  for(int i=0; i<N; i++){
    for(int j=0; j<M; j++){
      x = (((double)rand()-(RAND_MAX/2.0))/(RAND_MAX/2.0));
      gsl_matrix_set(matrix, i, j, x);
    }
  }
}

void print_matrix(gsl_matrix * matrix){
  // Prints a matrix
  int N = matrix->size1, M = matrix->size2;
  double x;
  for(int i=0; i<N; i++){
    for(int j=0; j<M; j++){
      x = gsl_matrix_get(matrix, i, j);
      printf("%lf ", x);
    }
    printf("\n");
  }
}

gsl_matrix * transpose(gsl_matrix * matrix){
  // Returns transpose of a matrix as a separate matrix
  gsl_matrix * matrix_T = gsl_matrix_alloc(matrix->size2, matrix->size1);
  for(int i=0; i< (matrix_T->size1); i++){
    for(int j=0; j< (matrix_T->size2); j++){
      gsl_matrix_set(matrix_T, i, j, gsl_matrix_get(matrix, j, i));
    }
  }
  return matrix_T;
}

void orthonormalize(gsl_matrix * matrix){
  const size_t M = matrix->size1, N = matrix->size2;
  //const size_t k =  GSL_MIN (M, N)
  int flag;
  gsl_vector * tau = gsl_vector_calloc(GSL_MIN (M, N));
  flag = gsl_linalg_QR_decomp(matrix, tau);
  if (flag==1)
    printf("Error!!: Unable to orthonormalize; fn: orthonormalize\n");

  gsl_matrix * Q = gsl_matrix_alloc(M, M);
  gsl_matrix * R = gsl_matrix_alloc(M, N);
  flag = gsl_linalg_QR_unpack(matrix, tau, Q, R);
  gsl_vector_free(tau);
  if (flag==1)
    printf("Error!!: Unable to unpack; fn: orthonormalize\n");
  gsl_matrix_free(R);
  gsl_vector * temp = gsl_vector_alloc(M);

  for(int i=0; i<N; i++){
    flag = gsl_matrix_get_col(temp, Q, i);
    if (flag==1)
      printf("Error!!: Unable to copy column from matrix; fn: orthonormalize\n");
    flag = gsl_matrix_set_col(matrix, i, temp);
    if (flag==1)
      printf("Error!!: Unable to copy column to matrix; fn: orthonormalize\n");
  }

  gsl_vector_free(temp);
  gsl_matrix_free(Q);
}

int matrix_mul(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C){
  // Multiplies A and B in the usual fashion and sets C = AB
  // returns 0 or success and 1 otherwise
  if(A->size2 != B->size1){
    printf("Error: Dimensions of A and B do not match; fn: matrix_mul\n");
    return 1;
  }
  else if(A->size1 != C->size1){
    printf("Error: Dimensions of A and C do not match; fn: matrix_mul\n");
    return 1;
  }
  else if(B->size2 != C->size2){
    printf("Error: Dimensions of B and C do not match; fn: matrix_mul\n");
    return 1;
  }
  double temp;
  for(int i=0; i< (C->size1); i++){
    for(int j=0; j< (C->size2); j++){
      temp = 0;
      for(int k=0; k< (B->size1); k++){
        temp = temp + (gsl_matrix_get(A, i, k)*gsl_matrix_get(B, k, j));
      }
      //printf("%lf ", temp);
      gsl_matrix_set(C, i, j, temp);
    }
    //printf("\n");
  }
  return 0;
}

int matrix_diagonal_product(gsl_vector * V, gsl_matrix * matrix, int type){
  /* Multiplies a diogonal matrix and diagonal matrix in O(NM) MxN is the dimension of the matrix
  Product overwrites matrix
  V represents a diagonal matrix in vector form
  type = 1 : mat(V) x matrix
  type = 2 : matrix x mat(V)
  */
  double temp;
  if(type==1){
    if(V->size != matrix->size1){
      printf("Error: Dimensions do not match, fn: matrix_diagonal_product\n");
      return 1;
    }
    for(int i=0; i< (matrix->size1); i++){
      for(int j=0; j< (matrix->size2); j++){
        temp = gsl_matrix_get(matrix, i, j);
        temp = temp*gsl_vector_get(V, i);
        gsl_matrix_set(matrix, i, j, temp);
      }
    }
  }
  else if(type==2){
    if(V->size != matrix->size2){
      printf("Error: Dimensions do not match, fn: matrix_diagonal_product\n");
      return 1;
    }
    for(int j=0; j< (matrix->size2); j++){
      for(int i=0; i< (matrix->size1); i++){
        temp = gsl_matrix_get(matrix, i, j);
        temp = temp*gsl_vector_get(V, j);
        gsl_matrix_set(matrix, i, j, temp);
      }
    }
  }
  else{
    printf("Error: Incorrect type of multiplication specified, fn: matrix_diagonal_product\n");
    return 1;
  }
  return 0;
}

gsl_matrix * pseudo_inverse(gsl_matrix * matrix){
  int flag, M = matrix->size1, N = matrix->size2;
  double temp;
  gsl_matrix * A = gsl_matrix_alloc(M, N);

  flag = gsl_matrix_memcpy(A, matrix);
  if (flag==1)
    printf("Error!!: Unable to copy matrix entries, fn:pseudo-inverse\n");

  gsl_vector * work = gsl_vector_alloc(N);
  gsl_vector * S = gsl_vector_alloc(N);
  gsl_matrix * V = gsl_matrix_alloc(N, N);
  // A = U x Sigma x V_T . The below function overwrites A with U. S = diag(Sigma). V is the transpose of V Transpose
  flag = gsl_linalg_SV_decomp(A, V, S, work);
  if (flag==1)
    printf("Error!!: Unable to get SVD decomposition, fn:pseudo-inverse\n");
  // Inverse of Sigma Matrix
  for(int i=0; i<N; i++){
    temp = gsl_vector_get(S, i);
    if(temp!=0)
      gsl_vector_set(S, i, 1.0/temp);
  }
  flag = matrix_diagonal_product(S, V, 2);
  if (flag==1)
    printf("Error!!: Unable to get matrix diagonal product, fn:pseudo-inverse\n");
  gsl_matrix * C = gsl_matrix_alloc(N, M);
  gsl_matrix * A_T = transpose(A);
  gsl_matrix_free(A);
  flag = matrix_mul(V, A_T, C);
  if (flag==1)
    printf("Error!!: Unable to multiply matrices, fn:pseudo-inverse\n");
  gsl_vector_free(S);
  gsl_vector_free(work);
  gsl_matrix_free(A_T);
  gsl_matrix_free(V);
  return C;
}

gsl_matrix * Generate_Matrix(int M, int N){
  // Returns a random matrices with orthonormal columns
  gsl_matrix * matrix = gsl_matrix_alloc(M, N);
  initialize_matrix(matrix);
  orthonormalize(matrix);
  return matrix;
}

int main(){
  int flag;
  for(int i=0; i<1; i++){
    gsl_matrix * A = Generate_Matrix(8, 3);
    printf("Matrix A\n");
    print_matrix(A);
    printf("\n");

    gsl_matrix * A_T = transpose(A);
    printf("Matrix A Transpose\n");
    print_matrix(A_T);
    printf("\n");
    gsl_matrix * product_ortho1 = gsl_matrix_alloc(8, 8);
    gsl_matrix * product_ortho2 = gsl_matrix_alloc(3, 3);
    flag = matrix_mul(A, A_T, product_ortho1);
    if (flag==1)
      printf("Error!!");
    flag = matrix_mul(A_T, A, product_ortho2);
    if (flag==1)
      printf("Error!!");
    flag++;
    printf("Orthonormal\n");
    print_matrix(product_ortho1);
    printf("\n");
    print_matrix(product_ortho2);
    printf("\n");
    gsl_matrix_free(product_ortho1);
    gsl_matrix_free(product_ortho2);

    product_ortho1 = gsl_matrix_alloc(8, 8);
    product_ortho2 = gsl_matrix_alloc(3, 3);
    gsl_matrix_free(A_T);
    gsl_matrix * inv = pseudo_inverse(A);
    printf("Inverse of Matrix A\n");
    print_matrix(inv);
    printf("\n");
    flag = matrix_mul(A, inv, product_ortho1);
    if (flag==1)
      printf("Error!!");
    flag = matrix_mul(inv, A, product_ortho2);
    if (flag==1)
      printf("Error!!");
    flag++;
    printf("Inverse\n");
    print_matrix(product_ortho1);
    printf("\n");
    print_matrix(product_ortho2);
    printf("\n");
    gsl_matrix_free(A);
    gsl_matrix_free(product_ortho1);
    gsl_matrix_free(product_ortho2);
    gsl_matrix_free(inv);
  }
}

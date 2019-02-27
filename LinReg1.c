#include<stdio.h>
#include<stdlib.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_blas.h>
#include <sys/times.h>
#include <unistd.h>
#include<time.h>
#include<math.h>

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

double frob_norm(gsl_matrix * matrix){
  double ans = 0;
  for(int i=0; i< (matrix->size1); i++){
    for(int j=0; j< (matrix->size2); j++){
      ans += (gsl_matrix_get(matrix, i, j)*gsl_matrix_get(matrix, i, j));
    }
  }
  return ans;
}

double Error(gsl_matrix * B, gsl_matrix * X, gsl_matrix * A, gsl_matrix * C){
  gsl_matrix * A_T = transpose(A);
  gsl_matrix * temp = gsl_matrix_alloc(B->size1, X->size2);
  gsl_matrix * temp1 = gsl_matrix_alloc(C->size1, C->size2);
  int flag;

  flag = matrix_mul(B, X, temp);
  if (flag==1)
    printf("Error!!; Unable to perform matrix multiplication; fn: Error\n");

  flag = matrix_mul(temp, A_T, temp1);
  if (flag==1)
    printf("Error!!; Unable to perform matrix multiplication; fn: Error\n");
  gsl_matrix_free(temp);

  flag = gsl_matrix_sub(temp1, C);
  if (flag==1)
    printf("Error!!: Dimension mismatch; fb: Error\n");

  double ans = frob_norm(temp1);//frob_norm(temp1);

  gsl_matrix_free(A_T);
  gsl_matrix_free(temp1);

  return ans;
}

gsl_matrix * Solve(gsl_matrix * B, gsl_matrix * A, gsl_matrix * C){
  gsl_matrix * inv_A = pseudo_inverse(A);
  gsl_matrix * inv_A_T = transpose(inv_A);
  gsl_matrix * inv_B = pseudo_inverse(B);
  gsl_matrix * temp = gsl_matrix_alloc(B->size2, C->size2);
  gsl_matrix * X = gsl_matrix_alloc(B->size2, A->size2);
  int flag=1;

  flag = matrix_mul(inv_B, C, temp);
  if (flag==1)
    printf("Error!!: Unable to multiply pinv(B) and C\n");
  flag = matrix_mul(temp, inv_A_T, X);
  if (flag==1)
    printf("Errorr!!: Unable to multiply pinv(B)C and pinv(A_T)\n");

  gsl_matrix_free(inv_A);
  gsl_matrix_free(temp);
  gsl_matrix_free(inv_B);
  gsl_matrix_free(inv_A_T);
  return X;
}

gsl_vector * Random_Diagonal(int M){
  gsl_vector * D = gsl_vector_alloc(M);
  srand(time(0));
  double x;
  for(int i=0; i<M; i++){
    x = (((double)rand()-(RAND_MAX/2.0))/(RAND_MAX/2.0));
    if(x>0)
      gsl_vector_set(D, i, 1);
    else
      gsl_vector_set(D, i, -1);
  }
  return D;
}

gsl_matrix * HadamardProduct(gsl_matrix * A, int N1, int N2){
  // N1 and N2 are inclusive
  // Assumming dimesnions of A are powers of 2
  if((N2-N1)==1){
    gsl_matrix * temp = gsl_matrix_alloc(2, A->size2);

    for(int j=0; j<(A->size2); j++){
      gsl_matrix_set(temp, 0, j, gsl_matrix_get(A, N1, j) + gsl_matrix_get(A, N1+1, j));
    }
    for(int j=0; j<(A->size2); j++){
      gsl_matrix_set(temp, 1, j, gsl_matrix_get(A, N1, j) - gsl_matrix_get(A, N1+1, j));
    }
    return temp;
  }
  int half = (N1+N2)/2, h1 = (N2-N1)/2;
  //printf("%d %d %d\n", N1, N2, half);
  gsl_matrix * temp1 = HadamardProduct(A, N1, half);
  gsl_matrix * temp2 = HadamardProduct(A, half+1, N2);
  //printf("Success\n");
  gsl_matrix * temp = gsl_matrix_alloc(N2-N1+1, A->size2);
  for(int i=0; i<=h1; i++){
    for(int j=0; j< (A->size2); j++){
      gsl_matrix_set(temp, i, j, gsl_matrix_get(temp1, i, j) + gsl_matrix_get(temp2, i, j));
    }
  }
  for(int i=h1 + 1; i<(N2-N1+1); i++){
    for(int j=0; j< (A->size2); j++){
      gsl_matrix_set(temp, i, j, gsl_matrix_get(temp1, i-h1 -1, j) - gsl_matrix_get(temp2, i-h1-1, j));
    }
  }

  gsl_matrix_free(temp1);
  gsl_matrix_free(temp2);

  return temp;
}

gsl_vector * sampling_matrix(int k, int N){
  gsl_vector * P = gsl_vector_alloc(k);
  int l;
  for(int i=0; i<k; i++){
    l = rand() % N;
    gsl_vector_set(P, i, l);
  }
  return P;
}

gsl_matrix * Subsampled(gsl_matrix * HDA, gsl_vector * P){
  int k = P->size;
  gsl_matrix * PHDA = gsl_matrix_alloc(k, HDA->size2);
  gsl_vector * temp = gsl_vector_alloc(HDA->size2);
  int l, flag;
  for(int i=0; i<k; i++){
    l = gsl_vector_get(P, i);
    flag = gsl_matrix_get_row(temp, HDA, l);
    if (flag==1)
      printf("Error!!: Unable to copy column from matrix; fn: Subsampled\n");
    flag = gsl_matrix_set_row(PHDA, i, temp);
    if (flag==1)
      printf("Error!!: Unable to copy column to matrix; fn: Subsampled\n");
  }
  gsl_vector_free(temp);

  return PHDA;
}

gsl_matrix * SRHT(int k, gsl_matrix * A1, gsl_matrix * B1, gsl_matrix * C1){
  gsl_vector * D1 = Random_Diagonal(A1->size1);
  gsl_vector * D2 = Random_Diagonal(B1->size1);

  gsl_matrix * A = gsl_matrix_alloc(A1->size1, A1->size2);
  gsl_matrix * B = gsl_matrix_alloc(B1->size1, B1->size2);
  gsl_matrix * C = gsl_matrix_alloc(C1->size1, C1->size2);
  int flag;

  flag = gsl_matrix_memcpy(A, A1);
  if (flag==1)
    printf("Error!!: Unable to copy matrix entries, fn:SRHT\n");
  flag = gsl_matrix_memcpy(B, B1);
  if (flag==1)
    printf("Error!!: Unable to copy matrix entries, fn:SRHT\n");
  flag = gsl_matrix_memcpy(C, C1);
  if (flag==1)
    printf("Error!!: Unable to copy matrix entries, fn:SRHT\n");

  flag = matrix_diagonal_product(D1, A, 1);
  if (flag==1)
    printf("Error!!: Issues in matrix multiplication of D1 and A\n");

  flag = matrix_diagonal_product(D2, B, 1);
  if (flag==1)
    printf("Error!!: Issues in matrix multiplication of D2 and B\n");

  flag = matrix_diagonal_product(D2, C, 1);
  if (flag==1)
    printf("Error!!: Issues in matrix multiplication of D2 and C\n");
  //printf("E1\n");
  gsl_matrix * HDA = HadamardProduct(A, 0, A->size1 - 1);
  //printf("E1a\n");
  gsl_vector * P1 = sampling_matrix(k, A->size1);
  //printf("E1b\n");
  gsl_matrix * PHDA = Subsampled(HDA, P1);
  gsl_matrix_free(HDA);
  //printf("E2\n");

  gsl_matrix * HDB = HadamardProduct(B, 0, B->size1-1);
  gsl_vector * P2 = sampling_matrix(k, B->size1);
  gsl_matrix * PHDB = Subsampled(HDB, P2);
  gsl_matrix_free(HDB);
  //printf("E3\n");

  gsl_matrix * HDC = HadamardProduct(C, 0, C->size1-1);
  gsl_matrix * PHDC = Subsampled(HDC, P2);
  gsl_matrix_free(HDC);
  gsl_matrix * PHDC_T = transpose(PHDC);
  gsl_matrix_free(PHDC);
  //printf("E4\n");
  flag = matrix_diagonal_product(D1, PHDC_T, 1);
  if (flag==1)
    printf("Error!!: Issues in matrix multiplication of D1 and PHDC_T; fn:SRHT:\n");
  gsl_matrix * HD_PHDC_T = HadamardProduct(PHDC_T, 0, PHDC_T->size1-1);
  gsl_matrix_free(PHDC_T);
  gsl_matrix * PHD_PHDC_T = Subsampled(HD_PHDC_T, P1);
  gsl_matrix_free(HD_PHDC_T);
  gsl_matrix * RCS_T = transpose(PHD_PHDC_T);
  gsl_matrix_free(PHD_PHDC_T);
  //printf("E5\n");
  gsl_matrix * X = Solve(PHDB, PHDA, RCS_T);
  //printf("E6\n");

  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_matrix_free(C);

  gsl_vector_free(D1);
  gsl_vector_free(D2);
  gsl_vector_free(P1);
  gsl_vector_free(P2);

  return X;
}

int main (void){
  //int m_arr[10] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32786, 65536};
  //int d_arr[7] = {5, 10, 25, 50, 75, 100, 125};
  int m_arr[6] = {128, 256, 512, 1024};
  int d_arr[4] = {16, 32, 64, 128};
  int m, d;
  double err=0, err1, tick_sec = sysconf(_SC_CLK_TCK);
  printf("The clock Tick rate is %lf\n", tick_sec);
  for(int i1=0; i1<4; i1++){
    m = m_arr[i1];
    for(int j1=0; j1<4; j1++){
      d = d_arr[j1];

      gsl_matrix * A = Generate_Matrix(m, d);
      gsl_matrix * B = Generate_Matrix(m, d);
      gsl_matrix * C = Generate_Matrix(m, m);
      struct tms  start, end;


      times(&start);
      gsl_matrix * X_true = Solve(B, A, C);
      times(&end);
      clock_t usr_time = end.tms_utime - start.tms_utime;
      clock_t sys_time = end.tms_stime - start.tms_stime;
      //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
      err = Error(B, X_true, A, C);
      printf("Error for exact solution is %lf and time taken is %lf ms\n", err, 1000*(usr_time + sys_time)/tick_sec);

      double Cons = 1.0, eps = 0.9, delta = 0.99;
      int k = (int)Cons*(d/(eps*eps))*log(d/delta)*log((d*m)/delta);
      printf("m: %d d: %d, k: %d\n",m,d,k);
      struct tms  start1, end1;
      times(&start1);
      gsl_matrix * X_approx = SRHT(k, A, B, C); //gsl_matrix_alloc(m, d);//
      times(&end1);
      clock_t usr_time1 = end1.tms_utime - start1.tms_utime;
      clock_t sys_time1 = end1.tms_stime - start1.tms_stime;
      //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
      err1 = Error(B, X_approx, A, C);
      printf("Error for approximate solution is %lf and time taken is %lf ms\n", err1, 1000*(usr_time1 + sys_time1)/tick_sec);

      gsl_matrix_free(A);
      gsl_matrix_free(B);
      gsl_matrix_free(C);
      gsl_matrix_free(X_true);
      gsl_matrix_free(X_approx);
      printf("\n");
    }

  }
  return 0;
}

/*******************************************************************************
* Copyright 2009-2016 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/
#include <stdlib.h>
#include "stdio.h"
#include "mkl.h"
#include "omp.h"
#include "math.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda );

int main()
{
    int thread=omp_get_max_threads();
    omp_set_num_threads(thread);

    MKL_INT  m = 3;
    MKL_INT  n = 4;
    MKL_INT  k = min(m,n);
    MKL_INT  lda = m; //column-major, n for row-major
    MKL_INT  ldu = m; //column-major, min(m,n) for row-major
    MKL_INT  ldvt = k; //column-major, n for row-major
    MKL_INT  lwork;
    MKL_INT info;
    double wkopt;
    double * work;
    char jobu='S';
    char jobvt= 'S';
    double a[]={
    	1, 5, 3,
    	2, 6, 4,
    	3, 7, 5,
    	4, 8, 6
    };
    double* s=(double*)malloc(k*sizeof(double));
    double* u = (double*)malloc(ldu*k*sizeof(double));
    double* vt = (double*)malloc(ldvt*n*sizeof(double));
    /* Executable statements */
    printf( " DGESVD Example Program Results\n" );
    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgesvd( &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, &info );
    lwork = (MKL_INT)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    /* Compute SVD */
    dgesvd( &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info );
    /* Check for convergence */
    if( info > 0 ) {
        printf( "The algorithm computing SVD failed to converge.\n" );
        exit( 1 );
    }
    /* Print singular values */
    print_matrix( "Singular values", 1, k, s, 1 );
    print_matrix( "Left singular vectors (stored columnwise)", ldu, k, u, ldu );
    //u=(s^-1)*U
    MKL_INT incx=1;
    #pragma omp parallel for
    for(int i=0; i<k; i++)
    {
       double ss;
       if(s[i] > 1.0e-9)
          ss=1.0/s[i];
       else
          ss=s[i];
       dscal(&m, &ss, &u[i*m], &incx);
     }
     print_matrix( "u=s'*U (after ?scal)", ldu, k, u, ldu );
     //inv(A)=(Vt)^T *u^T
     double* inva = (double*)malloc(n*m*sizeof(double));
     double alpha=1.0, beta=0.0;
     MKL_INT ld_inva=n;
     dgemm( "T", "T", &n, &m, &k, &alpha, vt, &ldvt, u, &ldu, &beta, inva, &ld_inva);
     print_matrix( "PseudoInverse of A", n, m, inva, ld_inva );
     return 0;
}

void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda ) {
    printf( "\n %s\n", desc );
    for( int i = 0; i < m; i++ ) {
        for( int j = 0; j < n; j++ ) printf( " %6.4f", a[i+j*lda] );
        printf( "\n" );
    }
}
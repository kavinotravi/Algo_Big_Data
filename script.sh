gcc -Wall -I/home/ravi/gsl/include -c basic_matrix.c
gcc -static -L/home/ravi/gsl/lib basic_matrix.o -lgsl -lgslcblas -lm
./a.out

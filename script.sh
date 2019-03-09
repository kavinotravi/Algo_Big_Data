gcc -Wall -DHAVE_INLINE -I/home/ravi/gsl/include -c LinReg1.c
gcc -static -L/home/ravi/gsl/lib LinReg1.o -lgsl -lgslcblas -lm
./a.out

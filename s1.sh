gcc -Wall -std=c99 -I/users/math/bs/kravi/gsl/include -c LinReg1.c
gcc -static -L/users/math/bs/kravi/gsl/lib LinReg1.o -lgsl -lgslcblas -lm
./a.out

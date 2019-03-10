gcc -Wall -DHAVE_INLINE -I/home/ravi/gsl/include -c -g LinReg1.c
gcc -static -L/home/ravi/gsl/lib LinReg1.o -lgsl -lgslcblas -lm
valgrind --tool=massif --main-stacksize=13998389608 ./a.out

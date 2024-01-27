CUC=nvcc
CUFLAGS=-g -I./inc -arch=sm_80

CC=gcc
CFLAGS=-Wall -Werror -std=gnu99 -fopenmp -lrt -lm

nop : 
	@echo "Pour compiler un programme, faire: make <nom_du_programme>"

% : %.cu
	$(CUC) $(CUFLAGS) -o $@ $^

% : %.c
	$(CC) $(CFLAGS) -o $@ $^

clean :
	rm -f matmat matmat_omp

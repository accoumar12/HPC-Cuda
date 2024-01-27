#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

#define NB_ITER 1000

/** Retourne la différence (en secondes) entre deux timespec */
double get_delta(struct timespec begin, struct timespec end) {
	return end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) * 1e-9;
}

void printMatrix(float *matrix, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%f ", matrix[i * size + j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Usage: %s <n>\n", argv[0]);
		return -1;
	}
	int n = atoi(argv[1]);

	// Allocation et initialisation des matrices
	float *A = (float *)calloc(n * n, sizeof(float));
	float *B = (float *)calloc(n * n, sizeof(float));
	float *C = (float *)calloc(n * n, sizeof(float));

	struct timespec begin, end;
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
		    if (i == j) {
			A[i * n + j] = i;
		    }
		    B[i * n + j] = j;
		}
	}
	
	clock_gettime(CLOCK_REALTIME, &begin);
	#pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float sum = 0.0;
			for (int k = 0; k < n; k++) {
    				sum += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = sum;
		}
	}
	clock_gettime(CLOCK_REALTIME, &end);
	
	printf("Temps du calcul matriciel avec OpenMP: %lf s.\n\n", get_delta(begin, end));
	
	/*
	printMatrix(A, n);
	printMatrix(B, n);
	printMatrix(C, n);
	*/

	// Libération de la mémoire
	free(A);
	free(B);
	free(C);

	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// On notera BS directement dans le code plutôt qu'une valeur en dure
#define BS 16

void printMatrix(float *matrix, int size) {
	float *hostMatrix = (float *)malloc(size * size * sizeof(float));
	cudaMemcpy(hostMatrix, matrix, size * size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%0.1f ", hostMatrix[i * size + j]);
		}
		printf("\n");
	}
	free(hostMatrix);
}

/** Retourne la différence (en secondes) entre deux timespec */
double get_delta(struct timespec begin, struct timespec end) {
	return end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) * 1e-9;
}

__global__ void mat_init(float *A, float *B, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * n + j;
    
    if (i < n && j < n) {
        if (i == j) {
            A[idx] = i;
        }
        B[idx] = j;
    }
}

__global__ void mat_mat(float * A, float * B, float * C, int n) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = i * n + j;
	
	if (i < n && j < n) {
		float sum = 0.0;
		for (int k=0; k<n; k++) {
			sum += A[n*i + k] * B[n*k+ j];
		}
		C[idx] = sum;
	}
}

__global__ void matmat_s(float *A, float *B, float *C, int n) {
	// Déclaration de la mémoire partagée pour les blocs de A et B
	__shared__ float sharedA[BS][BS];
	__shared__ float sharedB[BS][BS];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = i * n + j;

        float sum = 0.0;

	// Nombre de blocs nécessaires pour couvrir la matrice A et B
	int numBlocks = (n + BS - 1) / BS;

	for (int block = 0; block < numBlocks; ++block) {
	// Chargement du bloc de A et B en mémoire partagée
	if (i < n && block * BS + threadIdx.x < n) {
		sharedA[threadIdx.y][threadIdx.x] = A[i * n + block * BS + threadIdx.x];
	} 

	if (j < n && block * BS + threadIdx.y < n) {
		sharedB[threadIdx.y][threadIdx.x] = B[(block * BS + threadIdx.y) * n + j];
	}
	__syncthreads();

	// Calcul du produit scalaire du bloc en mémoire partagée
	for (int k = 0; k < BS; ++k) {
		sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
	}
	}

	// Écriture du résultat dans la matrice C
	if (i < n && j < n) {
		C[idx] = sum;
	}
}

int main(int argc, char * argv[]) {
	if (argc < 2) {
		printf("USAGE: %s <n>\n", argv[0]);
	}
	int n = atoi(argv[1]);
	int nb_blocks = (n + BS - 1) / BS;
	dim3 block_size(BS, BS);
	dim3 grid_size(nb_blocks, nb_blocks);
	struct timespec begin_matmat, end_matmat;

	// Allocations mémoires ...
	float * g_A, * g_B, * g_C;
	cudaMalloc((void **)&g_A, n * n * sizeof(float));
	cudaMalloc((void **)&g_B, n * n * sizeof(float));
	cudaMalloc((void **)&g_C, n * n * sizeof(float));

	int nb_iter = 1;
	for (int i = 0; i < nb_iter; i++) {
		// Appels aux kernels mat_init et mat_mat
		mat_init<<<grid_size, block_size>>>(g_A, g_B, n);
		cudaDeviceSynchronize();
		
		/*
		printMatrix(g_A, n);
		printf("\n");
		printMatrix(g_B, n);
		printf("\n");
		*/
		
		clock_gettime(CLOCK_REALTIME, &begin_matmat);
		mat_mat<<<grid_size, block_size>>>(g_A, g_B, g_C, n);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_REALTIME, &end_matmat);
		printf("Temps du calcul matriciel sans mémoire partagée: %lf s.\n\n", get_delta(begin_matmat, end_matmat)); 
		
		clock_gettime(CLOCK_REALTIME, &begin_matmat);
		matmat_s<<<grid_size, block_size>>>(g_A, g_B, g_C, n);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_REALTIME, &end_matmat);
		printf("Temps du calcul matriciel avec mémoire partagée: %lf s.\n\n", get_delta(begin_matmat, end_matmat));

		
		
		//printMatrix(g_C, n);
		//printf("\n");
	}
	cudaFree(g_A);
	cudaFree(g_B);
	

	float * h_C = (float*) malloc(n*n*sizeof(float));
	// Copie de C -> h_C
	cudaMemcpy(h_C, g_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i=0; i < n; i++) {
		for (int j=0; j < n; j++) {
			float expected = i * j;
			if (h_C[i*n+j] != expected) {
				printf("h_C[%d][%d] = %f != %f\n", i, j, h_C[i*n+j], expected);
			}
		}
	}

	// Libération de la mémoire
	free(h_C);
	cudaFree(g_C);
}

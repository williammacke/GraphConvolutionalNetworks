#ifndef RANDOM_H_
#define RANDOM_H_

#include "curand.h"
#include "curand_kernel.h"
#include "linAlg/matrix.h"


__global__ void initCurand(curandState_t *state, unsigned long seed, size_t n);


template <class T>
__global__ void randomize_data(curandState_t *state, T* out, size_t n, float p) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		if (curand_uniform(&state[idx]) < p) {
			out[idx] = 0;
		}
		else {
			out[idx] = 1;
		}

	}
}

template <class T>
void randomize_mat(curandState_t *state, Matrix<T>& out, float p) {
	size_t n = out.getN()*out.getM();
	randomize_data<<<(n+TPB-1)/TPB, TPB>>>(state, out.getData(), n, p);
	auto err = cudaDeviceSynchronize();
	if (err) {
		throw err;
	}
}

#endif

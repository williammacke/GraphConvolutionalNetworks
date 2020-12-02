#include "linAlg/random.h"


__global__ void initCurand(curandState *state, unsigned long seed, size_t n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		curand_init(seed, idx, 0, &state[idx]);
	}
}

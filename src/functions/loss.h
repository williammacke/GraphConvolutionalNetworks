#ifndef LOSS_H_
#define LOSS_H_
#include "linAlg/matrix.h"
#include "functions.activations.h"
#include <math.h>

struct dcross_entropy_with_logits {
	static oneSub os;
	template <class T>
	Matrix<T>& operator()(cublasHandle_t handle, const Matrix<T>& y, const Matrix<T>& y_, Matrix<T>& out, float* tmp1, float* ones) {
		size_t n = y.getN()*y.getM();
		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y.getData(), tmp1, n, os);

		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y_.getData(), out.getData(), n, os);
		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(tmp1, out.getData(), out.getData(), n);

		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(y.getData(), y_.getData, tmp, n);
		float alpha = 1.0f;
		float beta = 0.0f;
		cublasSaxpy(handle, n, &alpha, tmp1, 1, out.getData(), 1);
		cublasSgemv(handle, CUBLAS_OP_N, y.getN(), y.getM(),
				&alpha, y.getData(), y.getN(),
				ones, 1, &beta, tmp1, 1);
		rowWiseMul<<<(n+TPB-1)/TPB, TPB>>>(out.getData(), tmp1, out.getData(), out.getN(), out.getM());
		return out;
	}
}


struct scalar_log {
	static float epsilon;
template <class T>
	__host__ __device__
	T operator()(const T& x) {
		return log(x+epsilon);
	}
};


struct cross_entropy_with_logits {
	static size_t cap_ones;
	static float* ones;
	static size_t cap_tmp;
	static float* tmp;
	static size_t cap_tmp2;
	static float* tmp2;
	static scalar_log sl;
	static oneSub os;
	static dcross_entropy_with_logits dc;
	template <class T>
	void reallocOnes(size_t nCap) {
		if (ones != nullptr) {
			cudaFree(ones);
		}
		cudaMalloc(&ones, nCap);
		float* hOnes = new float[nCap];
		for (int i = 0; i < nCap; ++i) {
			hOnes[i] = 1;
		}
		cudaMemcpy(ones, hOnes, sizeof(float)*nCap, cudaMemcpyHostToDevice);
		cap_ones = nCap;
		delete[] hOnes;
	}
	void reallocTmp(size_t nCap) {
		if (tmp != nullptr) {
			cudaFree(tmp);
		}
		cudaMalloc(&tmp, nCap);
		cap_tmp = nCap;
	}

	void reallocTmp2(size_t nCap) {
		if (tmp2 != nullptr) {
			cudaFree(tmp2);
		}
		cudaMalloc(&tmp2, nCap);
		cap_tmp2 = nCap;
	}
	T operator()(cublasHandle_t handle, const Matrix<T>& y, const Matrix<T>& y_) {
		//TODO: Implement
		size_t n = y.getN()*y.getM();
		if (cap_tmp < n) {
			reallocTmp(y.getN()*y.getM());
		}
		if (cap_tmp2 < n) {
			reallocTmp2(n);
		}
		if (cap_ones < y_.getN()) {
			reallocOnes(y_.getN());
		}

		elementWiseAppy<<<(n+TPB-1)/TPB, TPB>>>(y_.getData(), tmp, n, os);
		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(tmp, tmp, n, sl);
		elementWiseAppy<<<(n+TPB-1)/TPB, TPB>>>(y.getData(), tmp2, n, os);
		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(tmp2, tmp2, n, sl);
		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(tmp, tmp2, tmp, n);


		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y_.getData(), tmp2, n, sl);
		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(y.getData(), tmp2, tmp2, n);

		float alpha=1.0f;
		float beta=0.0f;
		cublasSaxpy(handle, n, &alpha, tmp2, 1 tmp, 1);

		cublasSgemv(handle, CUBLAS_OP_N, y.getN(), y.getM(), &alpha, tmp, y_.getN(), ones, 1, &beta, tmp2, 1);
		cublasSgemv(handle, CUBLAS_OP_N, y.getN(), y.getM(), &alpha, y.getData(), y_.getN(), ones, 1, &beta, tmp, 1);
		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(tmp, tmp2, tmp, y.getN());

		cublasSdot(handle, y.getN(), ones, 1, tmp, 1, tmp2);
		float result;
		cudaMemcpy(&result, tmp2, sizeof(float), cudaMemcpyDeviceToHost);
		return result;
	}

	template <class T>
	Matrix<T>& derivative(cublasHandle_t handle, const Matrix<T>& y, const Matrix<T>& y_, Matrix<T>& out) {
		if (cap_ones < y.getN()) {
			reallocOnes(y.getN());
		}

		if (cap_tmp < y.getN()*y.getM()) {
			reallocTmp(y.getN()*y.getM());
		}
		return dc(handle, y, y_, out, tmp, ones);
	}

};


#endif

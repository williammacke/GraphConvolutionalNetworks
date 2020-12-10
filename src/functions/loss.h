#ifndef LOSS_H_
#define LOSS_H_
#include "linAlg/matrix.h"
#include "functions/activations.h"
#include <math.h>

struct oneSubE {
	__host__ __device__
	float operator()(const float x) const {
		return 1-x+epsilon;
	}
};


struct dcross_entropy_with_logits {
	static oneSubE ose;
	static oneSub os;
	static addE ae;
	template <class T>
	Matrix<T>& operator()(cublasHandle_t handle, const Matrix<T>& y, const Matrix<T>& y_, Matrix<T>& out, float* tmp1, float* ones) const {
		size_t n = y.getN()*y.getM();
		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y.getData(), tmp1, n, os);
		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y_.getData(), out.getData(), n, ose);
		cudaDeviceSynchronize();
		elementWiseDiv<<<(n+TPB-1)/TPB, TPB>>>(tmp1, out.getData(), out.getData(), n);
		cudaDeviceSynchronize();

		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y_.getData(), tmp1, n, ae);
		cudaDeviceSynchronize();
		elementWiseDiv<<<(n+TPB-1)/TPB, TPB>>>(y.getData(), tmp1, tmp1, n);
		cudaDeviceSynchronize();
		float alpha = -1;
		cublasSscal(handle, n, &alpha, tmp1, 1);
		cudaDeviceSynchronize();
		alpha = 1.0f;
		float beta = 0.0f;
		cublasSaxpy(handle, n, &alpha, tmp1, 1, out.getData(), 1);
		cudaDeviceSynchronize();
		cublasSgemv(handle, CUBLAS_OP_N, y.getN(), y.getM(),
				&alpha, y.getData(), y.getN(),
				ones, 1, &beta, tmp1, 1);
		cudaDeviceSynchronize();
		rowWiseMul<<<(n+TPB-1)/TPB, TPB>>>(out.getData(), tmp1, out.getData(), out.getN(), out.getM());
		cudaDeviceSynchronize();
		float total;
		cublasSdot(handle, y.getN(), tmp1, 1, ones, 1, &total);
		cudaDeviceSynchronize();
		total = 1.0f/total;
		//cublasSscal(handle, out.getN()*out.getM(), &total, out.getData(), 1);
		return out;
	}
};


struct scalar_log {
template <class T>
	__host__ __device__
	T operator()(const T& x) const {
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
	void reallocOnes(size_t nCap) {
		if (ones != nullptr) {
			cudaFree(ones);
		}
		auto err = cudaMalloc(&ones, nCap*sizeof(float));
		if (err) {
			throw err;
		}
		float* hOnes = new float[nCap];
		for (int i = 0; i < nCap; ++i) {
			hOnes[i] = 1;
		}
		err = cudaMemcpy(ones, hOnes, sizeof(float)*nCap, cudaMemcpyHostToDevice);
		if (err) {
			throw err;
		}
		cap_ones = nCap;
		delete[] hOnes;
	}
	void reallocTmp(size_t nCap) {
		if (tmp != nullptr) {
			cudaFree(tmp);
		}
		auto err = cudaMalloc(&tmp, nCap*sizeof(float));
		if (err) {
			throw err;
		}
		cap_tmp = nCap;
	}

	void reallocTmp2(size_t nCap) {
		if (tmp2 != nullptr) {
			cudaFree(tmp2);
		}
		auto err = cudaMalloc(&tmp2, nCap*sizeof(float));
		if (err) {
			throw err;
		}
		cap_tmp2 = nCap;
	}
	template <class T>
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

		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y_.getData(), tmp, n, os);
		cudaDeviceSynchronize();

		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(tmp, tmp, n, sl);
		cudaDeviceSynchronize();

		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y.getData(), tmp2, n, os);
		cudaDeviceSynchronize();

		//elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(tmp2, tmp2, n, sl);
		//cudaDeviceSynchronize();

		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(tmp, tmp2, tmp, n);
		cudaDeviceSynchronize();


		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(y_.getData(), tmp2, n, sl); 
		cudaDeviceSynchronize();
		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(y.getData(), tmp2, tmp2, n);
		cudaDeviceSynchronize();

		float alpha=1.0f;
		float beta=0.0f;
		cublasSaxpy(handle, n, &alpha, tmp2, 1, tmp, 1);
		cudaDeviceSynchronize();

		cublasSgemv(handle, CUBLAS_OP_N, y.getN(), y.getM(), &alpha, tmp, y_.getN(), ones, 1, &beta, tmp2, 1);
		cudaDeviceSynchronize();
		cublasSgemv(handle, CUBLAS_OP_N, y.getN(), y.getM(), &alpha, y.getData(), y_.getN(), ones, 1, &beta, tmp, 1);
		cudaDeviceSynchronize();
		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(tmp, tmp2, tmp, y.getN());
		cudaDeviceSynchronize();

		cublasSdot(handle, y.getN(), ones, 1, tmp, 1, tmp2);
		cudaDeviceSynchronize();
		float result;
		auto err = cudaMemcpy(&result, tmp2, sizeof(float), cudaMemcpyDeviceToHost);
		if (err) {
			throw err;
		}
		return -1*result;
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

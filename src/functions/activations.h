#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_
#include "linAlg/matrix.h"
#include <math.h>

struct scalar_relu {
	template <class T>
	__host__ __device__
	T operator()(const T& x) {
		if (x > 0) {
			return x;
		}
		return T();
	}
};

struct scalar_drelu {
	template <class T>
	__host__ __device__
	T operator()(const T& x) {
		if (x > 0) {
			return 1;
		}
		return T();
	}
};

struct drelu {
	static scalar_drelu r;;
	template <class T>
	Matrix<T>& operator()(Matrix<T>& mat) {
		return matApply(mat, mat, r);
	}
};


struct relu {
	static scalar_relu r;
	static drelu dr;
	template <class T>
	Matrix<T>& operator()(cublasHandle_t handle, Matrix<T>& mat) {
		return matApply(mat, mat, r);
	}

	template <class T>
	Matrix<T>& derivative(cublasHandle_t handle, const Matrix<T>& l, Matrix<T>& out) {
		return dr(out);
	}


};

struct scalar_exp {
	template <class T>
	__host__ __device__
	T operator()(const T& x) {
		return exp(x);
	}
};

struct oneSub {
	template <class T>
	__host__ __device__
	T operator()(const T& x) {
		return T(1) - x;
	}
};

struct dsoftmax {
	static oneSub os;
	template <class T>
	Matrix<T>& operator()(cublasHandle_t handle, const Matrix<T>& p, Matrix<T>& out, float *tmp) {
		size_t n = p.getN()*p.getM();
		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(p.getData(), tmp, n, os);
		elementWiesMul<<<(n+TPB-1)/TPB, TPB>>>(p.getData(), tmp, out.getData(), n);
		return out;
	}
};

struct softmax {
	static size_t cap_ones;
	static size_t cap_sum;
	static float* ones;
	static float* sum;
	static scalar_exp sexp;
	static dsoftmax ds;
	void reallocOnes(size_t newCap) {
		if (ones != nullptr) {
			cudaFree(ones);
		}
		cudaMalloc(&ones, newCap);
		float* hOnes = new float[newCap];
		for (int i =0; i < newCap; ++i) {
			hOnes[i] = 1.0f;
		}
		cudaMemcpy(ones, hOnes, sizeof(float)*newCap, cudaMemcpyHostToDevice);
		cap_ones = newCap;
		delete[] hOnes;
	}
	void reallocSum(size_t newCap) {
		if (sum != nullptr) {
			cudaFree(sum);
		}
		cudaMalloc(&sum, newCap);
		cap_sum = newCap;
	}
	template <class T>
	Matrix<T>& operator()(cublasHandle_t handle, Matrix<T>& mat) {
		if (cap_ones < mat.getN()) {
			reallocOnes(mat.getN());
		}
		if (cap_sum < mat.getN()) {
			reallocSum(mat.getN());
		}
		matApply(mat, mat, sexp);
		float alpha = 1.0f;
		float beta = 0.0f;
		cubasSgemv(handle, CUBLAS_OP_N, mat.getN(), mat.getM(), &alpha,
				mat.getData(), mat.getN(), ones, 1, &beta,
				sum, 1);
		size_t n = mat.getN()*mat.getM();
		rowWiseDiv<<<(n+TPB-1)/TPB, TPB>>>(mat.getData(), sum, mat.getData(), mat.getN(), mat.getM());
		return mat;

	}

	template <class T>
	Matrix<T>& derivative(cublasHandle_t handle, const Matrix<T>& p, Matrix<T>& out) {
		if (cap_sum < mat.getN()*mat.getM()) {
			reallocSum(mat.getN()*mat.getM());
		}
		return ds(handle, p, out, sum);
	}
};

#endif

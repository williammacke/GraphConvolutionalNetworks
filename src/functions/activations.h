#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_
#include "linAlg/matrix.h"
#include <math.h>
const  __device__ float epsilon = 0.00000000001f; 

struct addE {
	__host__ __device__
	float operator()(const float x) const {
		return x+epsilon;
	}
};

struct scalar_relu {
	__host__ __device__
	float operator()(const float& x) const {
		if (x > 0) {
			return x;
		}
		return 0;
	}
};

struct scalar_drelu {
	__host__ __device__
	float operator()(const float& x) const {
		if (x > 0) { return 1;
		}
		return 0;
	}
};

struct drelu {
	static scalar_drelu r;
	template <class T>
	Matrix<T>& operator()(const Matrix<T>& d, Matrix<T>& mat) const {
		return matApply(d, mat, r);
	}
};


struct relu {
	static scalar_relu r;
	static drelu dr;
	template <class T>
	Matrix<T>& operator()(cublasHandle_t handle, Matrix<T>& mat) const {
		return matApply(mat, mat, r);
	}

	template <class T>
	Matrix<T>& derivative(cublasHandle_t handle, const Matrix<T>& d, const Matrix<T>& l, Matrix<T>& out) const {
		return dr(d, out);
	}


};

struct scalar_exp {
	template <class T>
	__host__ __device__
	T operator()(const T& x) const {
		return exp(x);
	}
};

struct oneSub {
	template <class T>
	__host__ __device__
	T operator()(const T& x) const {
		return T(1) - x;
	}
};

struct dsoftmax {
	static oneSub os;
	template <class T>
	Matrix<T>& operator()(cublasHandle_t handle, const Matrix<T>& d, const Matrix<T>& p, Matrix<T>& out, float *tmp) const {
		size_t n = p.getN()*p.getM();
		elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(p.getData(), tmp, n, os);
		cudaDeviceSynchronize();
		elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(p.getData(), tmp, out.getData(), n);
		cudaDeviceSynchronize();
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
	static addE ae;
	void reallocOnes(size_t newCap) {
		if (ones != nullptr) {
			cudaFree(ones);
		}
		cudaError_t err;
		err = cudaMalloc(&ones, sizeof(float)*newCap);
		if (err) {
			std::cout << "Error malloc: " << err << std::endl;
			throw err;
		}
		float* hOnes = new float[newCap];
		for (int i =0; i < newCap; ++i) {
			hOnes[i] = 1.0f;
		}
		err = cudaMemcpy(ones, hOnes, sizeof(float)*newCap, cudaMemcpyHostToDevice);
		if (err) {
			std::cout << "Error: " << err << std::endl;
			throw err;
		}
		cap_ones = newCap;
		delete[] hOnes;
	}
	void reallocSum(size_t newCap) {
		if (sum != nullptr) {
			cudaFree(sum);
		}
		auto err = cudaMalloc(&sum, newCap*sizeof(float));
		if (err) {
			throw err;
		}
		cap_sum = newCap;
	}
	template <class T>
	Matrix<T>& operator()(cublasHandle_t handle, Matrix<T>& mat) {
		if (cap_sum < mat.getN()) {
			reallocSum(mat.getN());
		}
		if (cap_ones < mat.getM()) {
			std::cout << "allocating ones" << std::endl;
			reallocOnes(mat.getM());
		}
		matApply(mat, mat, sexp);
		cudaDeviceSynchronize();
		float alpha = 1.0f;
		float beta = 0.0f;
		cublasSgemv(handle, CUBLAS_OP_N, mat.getN(), mat.getM(), &alpha,
				mat.getData(), mat.getN(), ones, 1, &beta,
				sum, 1);
		cudaDeviceSynchronize();
		size_t n = mat.getN()*mat.getM();

		//elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(sum, sum, mat.getN(), ae);
		rowWiseDiv<<<(n+TPB-1)/TPB, TPB>>>(mat.getData(), sum, mat.getData(), mat.getN(), mat.getM());
		cudaDeviceSynchronize();
		return mat;

	}

	template <class T>
	Matrix<T>& derivative(cublasHandle_t handle, const Matrix<T>& d, const Matrix<T>& p, Matrix<T>& out) {
		if (cap_sum < p.getN()*p.getM()) {
			reallocSum(p.getN()*p.getM());
		}
		return ds(handle, d, p, out, sum);
	}
};

#endif

#ifndef MATRIX_H_
#define MATRIX_H_
#include <type_traits>
#include <cublas_v2.h>
#include <iostream>


const size_t TPB = 256;

template <class T>
class Matrix {
public:
	Matrix(int n, int m);
	~Matrix();
	void setValues(const T* vals);
	void gpuSetValues(const T* vals);
	T* getData();
	const T* getData() const;
private:
	T* data;
	int n, m;
public:
	size_t getN() const {
		return n;
	}
	size_t getM() const {
		return m;
	}
};


template <class T>
Matrix<T>::Matrix(int n, int m) :n(n), m(m) {
	cudaError_t error;
	error = cudaMalloc(&data, sizeof(T)*n*m);
	if (error != cudaSuccess) {
		std::cout << "malloc failed" << std::endl;
		throw error;
	}
}

template <class T>
Matrix<T>::~Matrix() {
	cudaFree(data);
}

template <class T>
void Matrix<T>::setValues(const T* vals) {
	auto err = cudaMemcpy(data, vals, sizeof(T)*n*m, cudaMemcpyHostToDevice);
	if (err) {
		throw err;
	}
}

template <class T>
void Matrix<T>::gpuSetValues(const T* vals) {
	auto err = cudaMemcpy(data, vals, sizeof(T)*n*m, cudaMemcpyDeviceToDevice);
	if (err) {
		throw err;
	}
}

template <class T>
T* Matrix<T>::getData() {
	return data;
}

template <class T>
const T* Matrix<T>::getData() const {
	return data;
}

Matrix<float>& matMul(cublasHandle_t handle, const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& out, bool transA=false, bool transB=false);
Matrix<float>& matMul_Add(cublasHandle_t handle, const Matrix<float>& A, const Matrix<float>& B, const Matrix<float>& C, Matrix<float>& out, bool transA=false,  bool transB=false);


Matrix<float>& add(cublasHandle_t handle, const Matrix<float>& A, Matrix<float>& out, float alpha);


template <class T>
__global__ void elementWiseMul(const T* data1, const T* data2, T* out, size_t n) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;

	if (idx < n) {
		out[idx] = data1[idx]*data2[idx];
	}
}

template <class T>
__global__ void elementWiseDiv(const T* data1, const T* data2, T* out, size_t n) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;

	if (idx < n) {
		out[idx] = data1[idx]/data2[idx];
	}
}


template <class T>
Matrix<T>& matElementMul(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& out) {
	size_t n = A.getN()*A.getM();

	elementWiseMul<<<(n+TPB-1)/TPB, TPB>>>(A.getData(), B.getData(), out.getData(), n);
	cudaDeviceSynchronize();
	return out;
}

template <class T>
__global__ void rowWiseMul(const T* data1, const T* data2, T* out, size_t r, size_t c) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	int rowidx = idx%r;
	size_t total = r*c;
	if (idx < total) {
		out[idx] = data1[idx] * data2[rowidx];
	}
}

template <class T>
__global__ void rowWiseDiv(const T* data1, const T* data2, T* out, size_t r, size_t c) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	int rowidx = idx%r;
	size_t total = r*c;
	if (idx < total) {
		out[idx] = data1[idx] / data2[rowidx];
	}
}


template <class T, class Op>
__global__ void elementWiseApply(const T* data, T* out, size_t n, const Op& op) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;

	if (idx < n) {
		out[idx] = op(data[idx]);
	}

}


template <class T, class Op>
Matrix<T>& matApply(const Matrix<T>& A, Matrix<T>& out, const Op& op) {
	size_t n = A.getN()*A.getM();

	std::cout << (n+TPB-1)/TPB << " " << TPB << std::endl;

	elementWiseApply<<<(n+TPB-1)/TPB, TPB>>>(A.getData(), out.getData(), n, op);
	cudaDeviceSynchronize();
	return out;
}


#endif

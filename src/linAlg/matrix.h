#ifndef MATRIX_H_
#define MATRIX_H_
#include <type_traits>

template <class T>
class Matrix {
public:
	static T* one;
	Matrix(int n, int m);
	~Matrix();
	void setValues(T* vals);
	void gpuSetValues(T* vals);
	const T* getData() const;
private:
	static size_t count;
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
static T* Matrix<T>::one = nullptr;
static size_t Matrix<T>::count = 0;

template <class T>
Matrix<T>::Matrix(int n, int m) :n(n), m(m) {
	cudaMalloc(&data, sizeof(T)*n*m);
	if (count == 0) {
		cudaMalloc(&one, sizeof(T));
		T tmp = T(1);
		cudaMemcpy(one, &tmp, sizeof(T), cudaMemcpyHostToDevice);
		//cudaMemCpy
	}
	++count;
}

template <class T>
Matrix<T>::~Matrix() {
	cudaFree(data);
	--count;
	if (count == 0) {
		cudaFree(one)
	}
}

template <class T>
void Matrix<T>::setValues(T* vals) {
	cudaMemcpy(data, vals, sizeof(T)*n*m, cudaMemcpyHostToDevice);
}

template <class T>
void Matrix<T>::gpuSetValues(T* vals) {
	cudaMemcpy(data, vals, sizeof(T)*n*m, cudaMemcpyDeviceToDevice);
}

template <class T>
const T* Matrix<T>::getData() const {
	return data;
}

template <class T>
Matrix<T>& matMul(cublasHandle_t handle, const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C, Matrix<T>& out) {
	out.gpuSetValues(C.getData());
	if constexpr(std::is_same_v<T, float>) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, A.getN(), B.getN(), A.getM(), 
				one, A.getData(), A.getN(), B.getData(), B.getN(), one, out.getData(), out.getN());

	}
	else if constexpr(std::is_same_v<T, double>) {
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, A.getN(), B.getN(), A.getM(), 
				one, A.getData(), A.getN(), B.getData(), B.getN(), one, out.getData(), out.getN());
	}
	else {
		static_assert(false, "Must be matrix of floats or doubles");
	}
}

#endif

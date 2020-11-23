#include "linAlg/matrix.h"



Matrix<float>& matMul(cublasHandle_t handle, const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& out) {
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.getN(), B.getM(), A.getM(), 
			&alpha, A.getData(), A.getN(), B.getData(), B.getN(), &beta, out.getData(), out.getN());
	return out;
}

Matrix<float>& matMul_Add(cublasHandle_t handle, const Matrix<float>& A, const Matrix<float>& B, const Matrix<float>& C, Matrix<float>& out) {
	float alpha = 1.0f;
	out.gpuSetValues(C.getData());
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.getN(), B.getM(), A.getM(), 
			&alpha, A.getData(), A.getN(), B.getData(), B.getN(), &alpha, out.getData(), out.getN());
	return out;
}

Matrix<float>& add(cublasHandle_t handle, const Matrix<float>& A, Matrix<float>& out, float alpha) {
	cublasSaxpy(handle, A.getN()*A.getM(), &alpha, A.getData(), 1, out.getData(), 1);
}

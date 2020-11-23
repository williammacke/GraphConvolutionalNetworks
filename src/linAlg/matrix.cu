#include "linAlg/matrix.h"



Matrix<float>& matMul(cublasHandle_t handle, const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& out, bool transA, bool transB) {
	auto opA = transA?CUBLAS_OP_T:CUBLAS_OP_N;
	auto opB = transB?CUBLAS_OP_T:CUBLAS_OP_N;
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSgemm(handle, opA, opB, transA?A.getM():A.getN(), transB?B.getN():B.getM(), transA?A.getN():A.getM(), 
			&alpha, A.getData(), transA?A.getM():A.getN(), B.getData(), transB?B.getM():B.getN(), &beta, out.getData(), out.getN());
	return out;
}

Matrix<float>& matMul_Add(cublasHandle_t handle, const Matrix<float>& A, const Matrix<float>& B, const Matrix<float>& C, Matrix<float>& out, bool transA, bool transB) {
	float alpha = 1.0f;
	auto opA = transA?CUBLAS_OP_T:CUBLAS_OP_N;
	auto opB = transB?CUBLAS_OP_T:CUBLAS_OP_N;
	out.gpuSetValues(C.getData());
	cublasSgemm(handle, opA, opB, transA?A.getM():A.getN(), transB?B.getN():B.getM(), transA?A.getN():A.getM(), 
			&alpha, A.getData(), transA?A.getM():A.getN(), B.getData(), transB?B.getM():B.getN(), &alpha, out.getData(), out.getN());
	return out;
}

Matrix<float>& add(cublasHandle_t handle, const Matrix<float>& A, Matrix<float>& out, float alpha) {
	cublasSaxpy(handle, A.getN()*A.getM(), &alpha, A.getData(), 1, out.getData(), 1);
}

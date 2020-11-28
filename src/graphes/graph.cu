#include "graphes/graph.h"


Matrix<float>& sparseMatMul(cusparseHandle_t handle, const Graph<float>& A, const Matrix<float>& B, Matrix<float>& out) {
	float alpha = 1.0f;
	float beta = 0.0f;
	cusparseSbsrmm(handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, A.getNumNodes(), A.getNumNodes(),
			A.getNumNodes(), A.getNNZB(), &alpha, A.getDescr(), A.getData(),
			A.getRowInd(), A.getColInd(), A.getBlockDim(), B.getData(), B.getN(), &beta, 
			out.getData(), out.getN());
	auto err = cudaDeviceSynchronize();
	if (err) {
		std::cout << "graph mult error: " << err << std::endl;
		throw err;
	}
	return out;
}

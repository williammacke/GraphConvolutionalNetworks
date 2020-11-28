#include "graphes/graph.h"


Matrix<float>& sparseMatMul(cusparseHandle_t handle, const Graph<float>& A, const Matrix<float>& B, Matrix<float>& out) {
	float alpha = 1.0f;
	float beta = 0.0f;
	auto err2 = cusparseSbsrmm(handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, A.getMB(), A.getNumNodes(),
			A.getMB(), A.getNNZB(), &alpha, A.getDescr(), A.getData(),
			A.getRowInd(), A.getColInd(), A.getBlockDim(), B.getData(), B.getN(), &beta, 
			out.getData(), out.getN());
	if (err2) {
		std::cout << "graph mult error2: " << err2 << std::endl;
		throw err2;
	}
	auto err = cudaDeviceSynchronize();
	if (err) {
		std::cout << "graph mult error: " << err << std::endl;
		throw err;
	}
	return out;
}

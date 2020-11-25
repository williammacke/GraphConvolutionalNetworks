#ifndef GRADIENT_DESCENT_H_
#define GRADIENT_DESCENT_H_

#include "linAlg/matrix.h"

//TODO: implement gradient descent optimizer

struct gradient_descent_optimizer {
	float lr;
	gradient_descent_optimizer(float lr) : lr(lr) { }
	template <class I, class Op>
	void optimize(cusparseHandle_t sHandle, cublasHandle_t bHandle,
			GCNLayer<I, Op>& layer, const Matrix<float>& d) {
		Matrix<float>& grad = layer.backward(bHandle, d);
		add(bHandle, grad, layer.getW(), lr);
		add(bHandle, layer.getD(), layer.getB(), lr);

	}
};

#endif

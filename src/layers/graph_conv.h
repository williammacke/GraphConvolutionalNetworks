#ifndef GRAPH_CONV_H_
#define GRAPH_CONV_H_
#include "linAlg/matrix.h"
#include "graphes/graph.h"
#include <string>


//TODO: implement graph convolutional layer
//
//
template <class I, class Op>
class GCNLayer {
public:
	GCNLayer(std::string name, size_t num_nodes, size_t num_inputs, size_t num_outputs, const Op& op, const I& init) : name(name), W(num_inputs, num_outputs), 
	B(num_nodes, num_outputs), d(num_nodes, num_outputs), init(init), op(op),
       XA(num_nodes, num_inputs), out(num_nodes, num_outputs)	{
		init.initialize(W, num_nodes, num_inputs, num_outputs);
		init.initialize(B, num_nodes, num_inputs, num_outputs);
	}
	~GCNLayer() { }

	Matrix<float>& forward(cusparseHandle_t sHandle, cublasHandle_t bHandle, 
			const Graph<float>& g, const Matrix<float>& in) {
		sparseMatMul(sHandle, g, in, XA);
		matMul_Add(bHandle, XA, W, B, d);
		out.gpuSetValues(d);
		op(out);
		return out;
	}

	Matrix<float>& backward(cublasHandle_t handle, const Matrix<float>& in, Matrix<float>& out, Matrix<float>& grad) {
		op.derivative(d);
		matElementMul(in, d, d);
		matMul(handle, XA, d, grad, true);
		matMul(handle, d, W, out, false, true);
		return grad;
	}


private:
	std::string name;
	Matrix<float> W;
	Matrix<float> B;
	Matrix<float> XA, out, d;
	I init;
	Op op;
};

#endif

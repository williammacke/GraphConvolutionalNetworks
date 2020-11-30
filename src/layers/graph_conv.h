#ifndef GRAPH_CONV_H_
#define GRAPH_CONV_H_
#include "linAlg/matrix.h"
#include "graphes/graph.h"
#include <string>

void printMat(const Matrix<float>& mat) {
	float* data = new float[mat.getN()*mat.getM()];
	cudaMemcpy(data, mat.getData(), sizeof(float)*mat.getN()*mat.getM(), cudaMemcpyDeviceToHost);
	for (int i = 0; i < mat.getN()*mat.getM(); ++i) {
		std::cout << data[i] << " ";
	}
	std::cout << std::endl;
	delete[] data;
}

//TODO: implement graph convolutional layer
//
//
template <class I, class Op>
class GCNLayer {
public:
	GCNLayer(std::string name, size_t num_nodes, size_t num_inputs, size_t num_outputs, const Op& op, const I& init) : name(name), W(num_inputs, num_outputs), 
	B(num_nodes, num_outputs), d(num_nodes, num_outputs), init(init), op(op),
       XA(num_nodes, num_inputs), out(num_nodes, num_outputs),
	grad(num_inputs, num_outputs), next(num_nodes, num_inputs){
		init.initialize(W, num_nodes, num_inputs, num_outputs);
		init.initialize(B, num_nodes, num_inputs, num_outputs);
	}
	~GCNLayer() { }

	Matrix<float>& forward(cusparseHandle_t sHandle, cublasHandle_t bHandle, 
			const Graph<float>& g, const Matrix<float>& in) {
		//XA.gpuSetValues(in.getData());
		sparseMatMul(sHandle, g, in, XA);
		cudaDeviceSynchronize();
		//std::cout << name << std::endl;
		//std::cout << "forward" << std::endl;
		//printMat(in);
		std::cout << "test1" << std::endl;
		//matMul(bHandle, XA, W, d);
		//cudaDeviceSynchronize();
		matMul_Add(bHandle, XA, W, B, d);
		out.gpuSetValues(d.getData());
		op(bHandle, out);
		//printMat(out);
		return out;
	}

	Matrix<float>& backward(cusparseHandle_t sHandle, cublasHandle_t handle, const Matrix<float>& in, const Graph<float>& g) {
		std::cout << "initial" << std::endl;
		//printMat(in);
		//printMat(W);
		//printMat(d);
		//printMat(out);
		op.derivative(handle, d, out, d);
		std::cout << "grad" << std::endl;
		//printMat(d);
		matElementMul(in, d, d);
		//printMat(d);
		matMul(handle, XA, d, grad, true);
		//printMat(grad);
		std::cout << "grad done" << std::endl;
		matMul(handle, d, W, next, false, true);
		//printMat(next);
		sparseMatMul(sHandle, g, next, next, true);
		//printMat(next);
		return grad;
	}

	Matrix<float>& getOut() {
		return out;
	}

	Matrix<float>& getNext() {
		return next;
	}

	Matrix<float>& getD() {
		return d;
	}

	Matrix<float>& getW() {
		return W;
	}
	
	Matrix<float>& getB() {
		return B;
	}


private:
	std::string name;
	Matrix<float> W;
	Matrix<float> B;
	Matrix<float> XA, out, d;
	Matrix<float> grad, next;
	I init;
	Op op;
};

#endif

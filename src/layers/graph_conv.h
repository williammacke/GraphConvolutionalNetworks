#ifndef GRAPH_CONV_H_
#define GRAPH_CONV_H_
#include "linAlg/matrix.h"
#include "graphes/graph.h"
#include <string>
#include "linAlg/random.h"
#include <cstdlib>

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
	GCNLayer(std::string name, size_t num_nodes, size_t num_inputs, size_t num_outputs,
			 const Op& op, const I& init, float dr=0.5) :
		name(name), W(num_inputs, num_outputs),
		num_inputs(num_inputs), num_outputs(num_outputs), num_nodes(num_nodes),
	B(num_nodes, num_outputs), d(num_nodes, num_outputs), init(init), op(op),
       XA(num_nodes, num_inputs), out(num_nodes, num_outputs),
	grad(num_inputs, num_outputs), next(num_nodes, num_inputs),
	DO(num_nodes, num_outputs), dr(dr) {
		init.initialize(W, num_nodes, num_inputs, num_outputs);
		init.initialize(B, num_nodes, num_inputs, num_outputs);
		size_t n = num_nodes*num_outputs;
		cudaMalloc(&randState, sizeof(curandState_t)*num_nodes*num_outputs);
		initCurand<<<(n+TPB-1)/TPB, TPB>>>(randState, rand(), num_nodes*num_outputs);
		auto err = cudaDeviceSynchronize();
		if (err) {
			throw err;
		}
	}
	~GCNLayer() { cudaFree(randState);}

	Matrix<float>& forward(cusparseHandle_t sHandle, cublasHandle_t bHandle, 
			const Graph<float>& g, const Matrix<float>& in, bool dropout = true) {
		//XA.gpuSetValues(in.getData());
		sparseMatMul(sHandle, g, in, XA);
		cudaDeviceSynchronize();
		//std::cout << name << std::endl;
		//std::cout << "forward" << std::endl;
		//printMat(in);
		std::cout << "test1" << std::endl;
		//matMul(bHandle, XA, W, d);
		//cudaDeviceSynchronize();
		if (dropout) {
			matMul_Add(bHandle, XA, W, B, d);
			out.gpuSetValues(d.getData());
			op(bHandle, out);
			randomize_mat(randState, DO, dr);
			matElementMul(out, DO, out);
			std::cout << "dropout " << dr << std::endl;
		}
		else {
			std::cout << "no dropout" << std::endl;
			grad.gpuSetValues(W.getData());
			if (dr > 0) {
				cublasSscal(bHandle, W.getN()*W.getM(),
						&dr, grad.getData(), 1);
				cudaDeviceSynchronize();
				cublasSscal(bHandle, B.getN()*B.getM(),
						&dr, out.getData(), 1);
				cudaDeviceSynchronize();
			}
			matMul_Add(bHandle, XA, grad, out, d);
			out.gpuSetValues(d.getData());
			op(bHandle, out);
		}
		//printMat(out);
		return out;
	}

	Matrix<float>& backward(cusparseHandle_t sHandle, cublasHandle_t handle, const Matrix<float>& in, const Graph<float>& g) {
		std::cout << "initial" << std::endl;
		//printMat(in);
		//printMat(W);
		//printMat(d);
		//printMat(out);
		//printMat(in);
		//std::cin.get();
		op.derivative(handle, d, out, d);
		std::cout << "grad" << std::endl;
		//printMat(d);
		matElementMul(DO, d, d);
		matElementMul(in, d, d);
		//printMat(d);
		matMul(handle, XA, d, grad, true);
		//printMat(grad);
		std::cout << "grad done" << std::endl;
		matMul(handle, d, W, next, false, true);
		//printMat(next);
		//std::cin.get();
		//sparseMatMul(sHandle, g, next, next, true);
		sparseMatMul(sHandle, g, next, next, false);
		//printMat(next);
		//std::cin.get();
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

	std::string getName() {
		return name;
	}

	size_t num_inputs, num_outputs;
	size_t num_nodes;
private:
	std::string name;
	Matrix<float> W;
	Matrix<float> B;
	Matrix<float> XA, out, d;
	Matrix<float> grad, next;
	Matrix<float> DO;
	float dr;
	curandState_t *randState;
	I init;
	Op op;
};

#endif

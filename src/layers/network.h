#ifndef NETWORK_H_
#define NETWORK_H_
#include <vector>
#include "layers/graph_conv.h"
#include <tuple>
#include <type_traits>
#include <iostream>


//TODO: implement final network class that applies a series of layers

template <class Loss, class Trainer, typename...Layers>
class Network {
public:
	Network(size_t num_nodes, size_t num_output_features, const Loss& loss, const Trainer& trainer, cublasHandle_t bhandle, cusparseHandle_t shandle, const Layers&...layers) : loss(loss), trainer(trainer), g(nullptr), bhandle(bhandle), shandle(shandle), ld(num_nodes, num_output_features), y(num_nodes, num_output_features), layers(layers...) {

	}
	~Network() { }
	void setGraph(Graph<float>* graph) {
		g = graph;
	}
	Matrix<float>& result(const Matrix<float>& in, bool dropout = true) {
		//std::cout << "layer 1" << std::endl;
		std::get<0>(layers).forward(shandle, bhandle, *g, in, dropout); 
		//return std::get<0>(layers).getOut();
		return result_helper<1>(in);
	}

	template <size_t I>
	typename std::enable_if<I == sizeof...(Layers) || (sizeof...(Layers) == 1), Matrix<float>&>::type result_helper(const Matrix<float>& in) {
		return std::get<I-1>(layers).getOut();
	}

	template <size_t I>
	typename std::enable_if<I < sizeof...(Layers) && (sizeof...(Layers) != 1), Matrix<float>&>::type result_helper(const Matrix<float>& in) {
		//std::cout << "layer " << (I+1) << std::endl;
		std::get<I>(layers).forward(shandle, bhandle, *g, std::get<I-1>(layers).getOut());
		return result_helper<I+1>(in);
	}

	void train(size_t num_epochs, const Matrix<float>& in) {
		constexpr size_t last = (sizeof...(Layers))-1;
		constexpr size_t current = last==0?0:last-1;
		for (int i = 0; i < num_epochs; ++i) {
			result(in);
			//std::cout << "train layer " << last << std::endl;
			loss.derivative(bhandle, y, std::get<last>(layers).getOut(), ld);
			trainer.optimize(shandle, bhandle, std::get<last>(layers), ld, *g);
			train_helper<current>(in);
		}
	}

	template <size_t I>
	typename std::enable_if<(0 < I) && sizeof...(Layers) != 1, void>::type train_helper(const Matrix<float>& in) {
		//std::cout << "train layer " << I << std::endl;
		trainer.optimize(shandle, bhandle, std::get<I>(layers), std::get<I+1>(layers).getNext(), *g);
		train_helper<I-1>(in);
	}

	template <size_t I>
	typename std::enable_if<I == 0 && sizeof...(Layers) != 1, void>::type train_helper(const Matrix<float>& in) {
		//std::cout << "train layer " << I << std::endl;
		trainer.optimize(shandle, bhandle, std::get<I>(layers), std::get<I+1>(layers).getNext(), *g);
	}
	template <size_t I>
	typename std::enable_if<I == 0 && sizeof...(Layers) == 1, void>::type train_helper(const Matrix<float>& in) {
		//std::cout << "train layer " << I << std::endl;
		//trainer.optimize(shandle, bhandle, std::get<I>(layers), std::get<I+1>(layers).getNext(), *g);
	}

	void setLabels(float* labels) {
		y.setValues(labels);
	}
	
	float getLoss() {
		constexpr size_t last = sizeof...(Layers)-1;
		return loss(bhandle, std::get<last>(layers).getOut(), y);
	}
private:
	std::tuple<Layers...> layers;
	Trainer trainer;
	Loss loss;
	cublasHandle_t bhandle;
	cusparseHandle_t shandle;
	Graph<float>* g;
	Matrix<float> y;
	Matrix<float> ld;

};


#endif

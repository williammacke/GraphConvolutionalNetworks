#ifndef NETWORK_H_
#define NETWORK_H_
#include <vector>


//TODO: implement final network class that applies a series of layers

template <class Loss, class Trainer>
class Network {
public:
	Network(const std::vector<GCNLayer>& layers, const Loss& loss, const Trainer& trainer, cublasHandle_t bhandle, cusparseHandle_t shandle) : layers(layers), loss(loss), trainer(trainer), g(nullptr), bhandle(bhandle), shandle(shandle) {

	}
	~Network() { }
	void setGraph(Graph* graph) {
		g = graph;
	}
	Matrix<float>& result(const Matrix<float>& in) {
		size_t num_layers = layers.size();
		layers[0].forward(shandle, bHandle, *g, in);
		for (int i = 1; i < num_layers; ++i) {
			layers[i].forward(shandle, bhandle, *g, layers[i-1].getOut());
		}
		return layers[num_layers-1].getOut();
	}
	void train(size_t num_epochs, const Matrix<float>& in) {
		size_t num_layers = layers.size();
		for (int i = 0; i < num_epochs; ++i) {
			result(in);
			for (int j = num_layers-1; j >= 0; --j) {
				trainer.optimize(sHandle, bHandle, layers[j]);
			}
		}
	}
private:
	std::vector<GCNLayer> layers;
	Trainer trainer;
	Loss loss;
	cublasHandle_t bhandle;
	cusparsehandle_t shandle;
	Graph* g;

};


#endif

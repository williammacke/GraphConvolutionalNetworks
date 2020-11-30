#include "layers/graph_conv.h"
#include "layers/network.h"
#include "functions/activations.h"
#include "initializer/random_normal.h"
#include "functions/loss.h"
#include "optimization/gradient_descent.h"





int main() {

	cublasHandle_t handle;
	cublasCreate(&handle);
	cusparseHandle_t sparseHandle;
	cusparseCreate(&sparseHandle);


	std::vector<std::vector<size_t>> adj_list = {{}, {2}, {1}};
	Graph<float> g(adj_list, sparseHandle);
	
	float labels[] = {1, 0, 0, 0, 1, 1};

	GCNLayer<random_normal_init, relu> layer1("l1", 3, 2, 2, relu(), random_normal_init(0, 0.01));
	GCNLayer<random_normal_init, softmax> layer2("l2", 3, 2, 2, softmax(), random_normal_init(0, 0.01));

	Network<cross_entropy_with_logits, gradient_descent_optimizer, GCNLayer<random_normal_init, relu>, GCNLayer<random_normal_init, softmax>> network(3, 2, {}, gradient_descent_optimizer(0.01f), handle, sparseHandle, layer1, layer2);
	//Network<cross_entropy_with_logits, gradient_descent_optimizer,  GCNLayer<random_normal_init, softmax>> network(3, 2, {}, gradient_descent_optimizer(0.1f), handle, sparseHandle, layer2);
	network.setGraph(&g);
	network.setLabels(labels);

	Matrix<float> input(3, 2);
	float data[] = {1, 0, 0, 0, 1, 2};
	input.setValues(data);

	for (int i = 0; i < 1000; i++) {
		printMat(network.result(input));
		std::cin.get();
		network.train(10, input);
	}

	printMat(network.result(input));
	std::cin.get();
	network.train(200, input);
	printMat(network.result(input));
	std::cin.get();
	network.train(200, input);
	printMat(network.result(input));
	std::cin.get();

	Matrix<float>& r = network.result(input);
	float result[6];
	cudaMemcpy(result, layer1.getOut().getData(), sizeof(float)*6, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 6; i++) {
		std::cout << result[i] << " ";
	}
	std::cout << std::endl;
	cudaMemcpy(result, r.getData(), sizeof(float)*6, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 6; i++) {
		std::cout << result[i] << " ";
	}
	std::cout << std::endl;




	cublasDestroy(handle);
	cusparseDestroy(sparseHandle);

	return 0;
}

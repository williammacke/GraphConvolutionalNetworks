#include "layers/graph_conv.h"
#include "layers/network.h"
#include "functions/activations.h"
#include "initializer/random_normal.h"
#include "functions/loss.h"
#include "optimization/gradient_descent.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusparseHandle_t sparseHandle;
	cusparseCreate(&sparseHandle); 
	std::ifstream content("../data/citeseer/citeseer.content");
	const int numPapers = 3312;
	const int numWords = 3703;
	std::vector<std::string> ids(numPapers);
	std::unordered_map<std::string, size_t> id_map;
	std::string label;
	std::unordered_map<std::string, int> label_map = {
		{"Agents", 0},
		{"AI", 1},
		{"DB", 2},
		{"IR", 3},
		{"ML", 4},
		{"HCI", 5}
	};

	float* data = new float[numPapers*numWords];
	float* labels = new float[numPapers*6];
	for (int i = 0; i < numPapers*6; ++i) {
		labels[i] = 0;
	}


	for (int i = 0; i < numPapers; ++i) {
		content >> ids[i];
		id_map[ids[i]] = i;
		for (int j = 0; j < numWords; ++j) {
			content >> data[j*numPapers+i];
		}
		content >> label;
		int li = label_map[label];
		labels[li*numPapers+i] = 1.0f;
	}

	content.close();
	std::vector<std::vector<size_t>> adj_list(numPapers);
	std::vector<std::unordered_set<size_t>> adj_set(numPapers);
	std::ifstream cites("../data/citeseer/citeseer.cites");


	std::string id1;
	std::string id2;
	for (int i = 0; i < numPapers; ++i) {
		cites >> id1;
		cites >> id2;

		int i1 = id_map[id1];
		int i2 = id_map[id2];

		if (adj_set[i1].find(i2) == adj_set[i1].end()) {
			adj_list[i1].push_back(i2);
			adj_set[i1].insert(i2);
		}
		if (adj_set[i2].find(i1) == adj_set[i2].end()) {
			adj_list[i2].push_back(i1);
			adj_set[i2].insert(i1);
		}
	}

	cites.close();


	Matrix<float> features(numPapers, numWords);
	features.setValues(data);

	Graph<float> g(adj_list, sparseHandle);
	std::cin.get();

	GCNLayer<random_normal_init, relu> layer1("l1", numPapers, numWords, 100, relu(),
			random_normal_init(0, 0.1));
	GCNLayer<random_normal_init, softmax> layer2("l2", numPapers, 100, 6, softmax(),
			random_normal_init(0, 0.1));


	Network<cross_entropy_with_logits, gradient_descent_optimizer, GCNLayer<random_normal_init, relu>, GCNLayer<random_normal_init, softmax>> network(numPapers, 6, {}, gradient_descent_optimizer(0.0001f), handle, sparseHandle, layer1, layer2);
	network.setGraph(&g);
	network.setLabels(labels);

	float* result = new float[numPapers*6];
	try {
	cudaMemcpy(result, network.result(features).getData(), sizeof(float)*6*numPapers, cudaMemcpyDeviceToHost);
	} catch(int i) { std::cout << "Error " << i << std::endl; }
	for (int i = 0; i < numPapers; ++i) {
		std::cout << ids[i] << " ";
		for (int j = 0; j < 6; j++) {
			std::cout << result[j*numPapers+i] << " ";
		}
		std::cout << std::endl;

	}
	for (int i = 0; i < 10; ++i) {
		std::cout << network.getLoss() << std::endl;
		std::cin.get();
		network.train(10, features);
	}

	delete[] data;
	delete[] labels;
	delete[] result;
	cublasDestroy(handle);
	cusparseDestroy(sparseHandle);
}

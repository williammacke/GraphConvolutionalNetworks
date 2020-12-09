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
#include <cstdlib>

size_t argmax(float* data, size_t n) {
	float m = data[0];
	size_t mi = 0;
	for (int i = 0; i < n; ++i) {
		if (data[i] > m) {
			m = data[i]; mi = i;
		} } return mi;

}

int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusparseHandle_t sparseHandle;
	cusparseCreate(&sparseHandle); 
	std::ifstream content("../data/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab");
	const int numPapers = 19717;
	const int numWords = 500;
	const int numEdges = 44338;
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
	float* labels = new float[numPapers*3];
	float* labels2 = new float[numPapers*3];
	int count[3];
	for (int i =0; i < 3; ++i) {
		count[i] = 0;
	}
	for (int i = 0; i < numPapers*3; ++i) {
		labels[i] = 0;
		labels2[i] = 0;
	}

	std::unordered_map<std::string, int> w_map;

	std::vector<size_t> test;
	std::string temp;
	getline(content, temp);
	content >> temp;
	for (int i = 0; i < numWords; ++i) {
		content >> temp;
		std::cout << temp << std::endl;
		int i1 = 0, i2 = 0;
		while (i1 < temp.length() && temp[i1] != ':') {
			++i1;
		}
		i2 = i1+1;
		while (i2 < temp.length() && temp[i2] != ':') {
			++i2;
		}
		std::cout << i1 << " " << i2 << std::endl;
		temp = temp.substr(i1+1, i2-(i1+1));
		w_map[temp] = i;
	}
	std::string temp2;
	getline(content, temp2);

	for (int i = 0; i < numPapers; ++i) {
		content >> ids[i];
		content >> label;
		std::cout << ids[i] << std::endl;
		std::cout << label << std::endl;

		int cindex = 0;
		while (cindex < label.length() && !isdigit(label[cindex])) {
			cindex++;
		}
		label = label.substr(cindex, label.length()-cindex);

		id_map[ids[i]] = i;
		float sum = 0.0f;
		content >> temp;
		while(temp.length() > 0 && temp[0] != 's') {
			std::cout << temp;
			cindex = 0;
			while (cindex < temp.length() && !isdigit(temp[cindex])) {
				cindex++;
			}
			std::cout << temp << std::endl;
			temp2 = temp.substr(cindex, temp.length()-cindex);
			temp = temp.substr(0, cindex-1);
			std::cout << temp << std::endl;
			std::cout << temp2 << std::endl;
			data[w_map[temp]*numPapers+i] = atof(temp2.c_str());
			sum += data[w_map[temp]*numPapers+i];
			content >> temp;
		}
		std::cout << "blargh" << std::endl;
		for (int j = 0; j < numWords; ++j) {
			data[j*numPapers+i] /= sum;
		}
		int li = atoi(label.c_str()) - 1;
		std::cout << label << std::endl;
		std::cout << li << std::endl;
		std::cout << "blargh 2" << std::endl;
		if (count[li] < 20) {
			labels[li*numPapers+i] = 1.0f;
			++count[li];
			//test.push_back(i);
		}
		else {
			labels2[li*numPapers+i] = 1.0f;
			test.push_back(i);
		}
		//getline(content, temp);
		//std::cin.get();
	}

	content.close();
	std::cin.get();
	std::vector<std::vector<size_t>> adj_list(numPapers);
	std::vector<std::unordered_set<size_t>> adj_set(numPapers);
	std::ifstream cites("../data/Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab");
	getline(cites, temp);
	getline(cites, temp);


	std::string id1;
	std::string id2;
	for (int i = 0; i < numEdges; ++i) {
		cites >> id1;
		cites >> id1;
		cites >> id2;
		cites >> id2;
		std::cout << id1 << std::endl;
		std::cout << id2 << std::endl;
		int cindex = 0;
		while (cindex < id1.length() && id1[cindex] != ':') {
			++cindex;
		}
		++cindex;
		id1 = id1.substr(cindex, id1.length()-cindex);
		cindex = 0;
		while (cindex < id2.length() && id2[cindex] != ':') {
			++cindex;
		}
		++cindex;
		id2 = id2.substr(cindex, id2.length()-cindex);

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

	GCNLayer<random_normal_init, relu> layer1("l1", numPapers, numWords, 32, relu(),
			random_normal_init(0, 0.01));
	GCNLayer<random_normal_init, softmax> layer2("l2", numPapers, 32, 3, softmax(),
			random_normal_init(0, 0.1), -1);


	Network<cross_entropy_with_logits, adam, GCNLayer<random_normal_init, relu>, GCNLayer<random_normal_init, softmax>> network(numPapers, 3, {}, adam(), handle, sparseHandle, layer1, layer2);
	network.setGraph(&g);
	network.setLabels(labels);

	float* result = new float[numPapers*3];
	try {
	cudaMemcpy(result, network.result(features).getData(), sizeof(float)*3*numPapers, cudaMemcpyDeviceToHost);
	} catch(int i) { std::cout << "Error " << i << std::endl; }
	for (int i = 0; i < numPapers; ++i) {
		std::cout << ids[i] << " ";
		for (int j = 0; j < 6; j++) {
			std::cout << result[j*numPapers+i] << " ";
		}
		std::cout << std::endl;

	}
	for (int i = 0; i < 20; ++i) {
		cudaMemcpy(result, network.result(features, false).getData(), sizeof(float)*3*numPapers, cudaMemcpyDeviceToHost);
		float l[3];
		float l_[3];
		float total = 0.0f;
		float correct = 0.0f;
		for (auto& t:test) {
			for (int j = 0; j < 3; ++j) {
				l[j] = labels2[j*numPapers+t];
				//l[j] = labels[j*numPapers+t];
				l_[j] = result[j*numPapers+t];
			}
			if (argmax(l, 3) == argmax(l_, 3)) {
				correct += 1;
			}
			total += 1;
		}
		std::cout << network.getLoss() << std::endl;
		network.setLabels(labels2);
		std::cout << network.getLoss() << std::endl;
		network.setLabels(labels);
		std::cout << "Acc: " << (correct/total) << std::endl;
		std::cin.get();
		network.train(10, features);
	}

	delete[] data;
	delete[] labels;
	delete[] result;
	cublasDestroy(handle);
	cusparseDestroy(sparseHandle);
}

#ifndef RANDOM_NORMAL_H_
#define RANDOM_NORMAL_H_

#include <random>
#include <chrono>
#include "linAlg/matrix.h"

struct random_normal_init {
	const float mean;
	const float std;
	random_normal_init(float mean, float std) : mean(mean), std(std) { }
	~random_normal_init() { }
	void initialize(Matrix<float>& mat, size_t num_nodes, size_t num_inputs, size_t num_outputs) const {
		float* data = new float[mat.getN()*mat.getM()];

		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  		std::default_random_engine generator (seed);
		std::normal_distribution<float> dist(mean, std);
		for (int i = 0; i < mat.getN()*mat.getM(); ++i) {
			data[i] = dist(generator);
		}

		mat.setValues(data);

		delete[] data;
	}
};


#endif

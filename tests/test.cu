#include "linAlg/matrix.h"
#include "graphes/graph.h"
#include "layers/graph_conv.h"
#include <iostream>
#include <vector>

struct addTwo {
	__host__ __device__
	float operator() (float a) const{
		return a+2.0f;
	}
};

int main() {
	Matrix<float> a(2,2);
	Matrix<float> b(2, 2);
	Matrix<float> c(2, 2);
	Matrix<float> d(2, 2);

	float data[] = {2,1,1,1};
	float data2[] = {0,0,0,0};
	a.setValues(data);
	b.setValues(data);
	c.setValues(data2);
	float blargh = 0.0f;


	cudaMemcpy(data2, a.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;
	data2[0] = 0;
	cudaMemcpy(data2, b.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;
	cudaMemcpy(data2, c.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;
	


	cublasHandle_t handle;
	cublasStatus_t status;
	cudaError_t error;

	status = cublasCreate(&handle);
	if (status != cudaSuccess) {
		std::cout << "error";
	}

	matMul(handle, a, b, d);
	cudaMemcpy(data2, d.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;


	matMul_Add(handle, a, b, c, d);
	cudaMemcpy(data2, d.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;


	add(handle, a, d, 0.5f);
	//cudaMemcpy(data2, d.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	error = cudaMemcpy(data2, d.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "error" << " " << error << std::endl;
	}
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;



	matElementMul(a, b, d);
	cudaMemcpy(data2, d.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;

	matApply(a, d, addTwo{});
	error = cudaMemcpy(data2, d.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "error" << " " << error << std::endl;
		if (error == cudaErrorInvalidValue) {
			std::cout << "invalid value" << std::endl;
		}

		if (error == cudaErrorInvalidValue) {
			std::cout << "invalid value" << std::endl;
		}

		if (error == cudaErrorInvalidMemcpyDirection) {
			std::cout << "invalid direction" << std::endl;
		}
	}
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;




	std::vector<std::vector<size_t>> adj_list = {{1}, {0,2}, {1}};


	cusparseHandle_t sparseHandle;
	cusparseCreate(&sparseHandle);

	Graph<float> g(adj_list, sparseHandle);

	Matrix<float> e(3, 3);
	float data3[] = {1,1,0,0,1,0,0,0,1};
	e.setValues(data3);
	Matrix<float> f(3, 3);

	
	sparseMatMul(sparseHandle, g, e, f);
	cudaMemcpy(data3, f.getData(), 9*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data3[0] << " " <<  data3[1] << " " << data3[2] << " " << data3[3] << " " << data3[4] << " " << data3[5] << " " << data3[6] << " " << data3[7] << " " << data3[8] << std::endl;


	Matrix<float> m1(2, 3);
	Matrix<float> m2(3, 4);
	Matrix<float> m3(2, 4);

	float data_m1[] = {1, 2, 3, 1, 2, 3};
	m1.setValues(data_m1);

	float data_m2[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
	m2.setValues(data_m2);



	float data_m4[8];

	matMul(handle, m1, m2, m3);
	cudaMemcpy(data_m4, m3.getData(), 8*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data_m4[0] << " " <<  data_m4[1] << " " << data_m4[2] << " " << data_m4[3] << " " << data_m4[4] << " " << data_m4[5] << " " << data_m4[6] << " " << data_m4[7] << std::endl;
	


	cublasDestroy(handle);
	cusparseDestroy(sparseHandle);
	return 0;
}

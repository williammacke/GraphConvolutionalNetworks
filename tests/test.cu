#include "linAlg/matrix.h"
#include <iostream>

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
	cudaMemcpy(&blargh, Matrix<float>::one, sizeof(float),cudaMemcpyDeviceToHost );
	std::cout << blargh <<std::endl;


	cudaMemcpy(data2, a.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;
	data2[0] = 0;
	cudaMemcpy(data2, b.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;
	cudaMemcpy(data2, c.getData(), 4*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << data2[0] << data2[1] << data2[2] << data2[3] << std::endl;
	std::cout << Matrix<float>::one << std::endl;
	


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




	cublasDestroy(handle);
	return 0;
}

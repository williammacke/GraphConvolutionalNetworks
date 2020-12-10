#ifndef GRADIENT_DESCENT_H_
#define GRADIENT_DESCENT_H_

#include <unordered_map>

#include "linAlg/matrix.h"

//TODO: implement gradient descent optimizer

struct square_root {
    __host__ __device__
    float operator() (float a) const {
        return std::sqrt(a);
    }
};

struct inverse {
    __host__ __device__
    float operator() (float a) const {
        return 1 / a;
    }
};


struct gradient_descent_optimizer {
	float lr;
	gradient_descent_optimizer(float lr) : lr(lr) { }
	template <class I, class Op>
	void optimize(cusparseHandle_t sHandle, cublasHandle_t bHandle,
			GCNLayer<I, Op>& layer, const Matrix<float>& d, const Graph<float>& g) {
		Matrix<float>& grad = layer.backward(sHandle, bHandle, d, g);
		add(bHandle, grad, layer.getW(), -lr);
		add(bHandle, layer.getD(), layer.getB(), -lr);

	}
};


struct AdamParameters {
    AdamParameters(size_t num_inputs, size_t num_outputs)
        : b1(float(0.9)),
          b2(float(0.999)),
          b1_t(float(0.9)),
          b2_t(float(0.999)),
          num_inputs(num_inputs),
          num_outputs(num_outputs),
          m(num_inputs, num_outputs),
          mHat(num_inputs, num_outputs),
          u(num_inputs, num_outputs),
          uHat(num_inputs, num_outputs),
          tmp(num_inputs, num_outputs),
          ones(num_inputs, num_outputs) {
        float ones_[num_inputs * num_outputs];
        float zeros[num_inputs * num_outputs];
        for (int i = 0; i < num_inputs * num_outputs; i++) {
            ones_[i] = 1;
            zeros[i] = 0;
        }
        ones.setValues(ones_);
        m.setValues(zeros);
        u.setValues(zeros);
    }

    size_t num_inputs, num_outputs;
    Matrix<float> m, mHat;
    Matrix<float> u, uHat;
    Matrix<float> tmp;
    Matrix<float> ones;

    float b1;     // decay term
    float b2;     // decay term
    float b1_t;   // decay term power t
    float b2_t;   // decay term power t
};


/**
 * Taken from tiny-dnn github, https://github.com/tiny-dnn/tiny-dnn
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
struct adam {
    adam()
        : alpha(float(0.01)),  // Same alpha as tkipf
          eps(float(1e-8)) {}

    void step(cublasHandle_t bHandle, Matrix<float>& grad,
              Matrix<float>& params, AdamParameters& adamParams) {
        // cubasScopy: docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-copy
        // cublasSscal: docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-scal
        Matrix<float>& m    = adamParams.m;
        Matrix<float>& mHat = adamParams.mHat;
        Matrix<float>& u    = adamParams.u;
        Matrix<float>& uHat = adamParams.uHat;
        Matrix<float>& tmp = adamParams.tmp;
        Matrix<float>& ones = adamParams.ones;

        // Scale m by b1
        cublasSscal(bHandle, m.getM() * m.getN(), &adamParams.b1, m.getData(), 1);
        // mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
        add(bHandle, grad, m, float(1) - adamParams.b1);

        // Store grad * grad in tmp
        matElementMul(grad, grad, tmp);
        // Scale u by b2
        cublasSscal(bHandle, u.getM() * u.getN(), &adamParams.b2, u.getData(), 1);
	cudaDeviceSynchronize();
        // ut[i] = b2 * ut[i] + (float_t(1) - b2) * dW[i] * dW[i];
        add(bHandle, tmp, u, float(1) - adamParams.b2);

        // Calculate
        // W[i] -= alpha * (mt[i] / (float_t(1) - b1_t)) /
        //         std::sqrt((vt[i] / (float_t(1) - b2_t)) + eps);

        // mHat = m / (1 - b1_t)
        float alpha_ = float(1) / (1 - adamParams.b1_t);
        cublasScopy(bHandle, m.getM() * m.getN(), m.getData(), 1, mHat.getData(), 1);
	cudaDeviceSynchronize();
        cublasSscal(bHandle, mHat.getM() * mHat.getN(), &alpha_, mHat.getData(), 1);
	cudaDeviceSynchronize();
        // uHat = u / (1 - b2_t)
        alpha_ = float(1) / (1 - adamParams.b2_t);
        cublasScopy(bHandle, u.getM() * u.getN(), u.getData(), 1, uHat.getData(), 1);
	cudaDeviceSynchronize();
        cublasSscal(bHandle, uHat.getM() * uHat.getN(), &alpha_, uHat.getData(), 1);
	cudaDeviceSynchronize();
        // tmp = m_hat / (sqrt(u_hat) + eps)
        // tmp = sqrt(uHat)
        matApply(uHat, tmp, square_root{});
        // tmp = tmp + eps
        add(bHandle, ones, tmp, eps);
        // tmp = tmp^(-1)
        matApply(tmp, tmp, inverse{});
        // tmp = mHat * tmp
        matElementMul(tmp, mHat, tmp);
        add(bHandle, tmp, params, -alpha);

        // Decay b1_t and b2_t
        adamParams.b1_t *= adamParams.b1;
        adamParams.b2_t *= adamParams.b2;
    }

    template <class I, class Op>
    void optimize(cusparseHandle_t sHandle, cublasHandle_t bHandle,
                  GCNLayer<I, Op>& layer, const Matrix<float>& d, const Graph<float>& g) {
        std::string weightName = layer.getName() + "_weights";
        std::string biasName = layer.getName() + "_bias";
        if (mapping.count(weightName) == 0) {
            mapping[weightName] = new AdamParameters(layer.num_inputs,
                                                     layer.num_outputs);
            mapping[biasName] = new AdamParameters(layer.num_nodes,
                                                   layer.num_outputs);
        }
        AdamParameters& weightParams = *mapping[weightName];
        Matrix<float>& grad = layer.backward(sHandle, bHandle, d, g);
        Matrix<float>& toApply = layer.getW();
        step(bHandle, grad, toApply, weightParams);

        // Simple gradient descent on biases does better than Adam
        add(bHandle, layer.getD(), layer.getB(), -alpha);
//      AdamParameters& biasParams = *mapping[biasName];
//      Matrix<float>& biasGrad = layer.getD();
//      Matrix<float>& a = layer.getB();
//      step(bHandle, biasGrad, a, biasParams);
        return;
    }

    float alpha;  // learning rate
    std::unordered_map<std::string, AdamParameters*> mapping;

private:
    float eps;  // constant value to avoid zero-division
};

#endif

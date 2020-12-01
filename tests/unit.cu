// unit.cu
// Sanity check basic cuda commands.

// Let Catch provide main():
#define CATCH_CONFIG_MAIN

#include <iostream>
#include <vector>

#include "catch.hpp"
#include "graphes/graph.h"
#include "layers/graph_conv.h"
#include "linAlg/matrix.h"


TEST_CASE("cudaMemcpy") {
    Matrix<float> A(2, 2);
    float in[]  = {2, 1, 1, 1};
    float out[] = {0, 0, 0, 0};
    A.setValues(in);
    cudaMemcpy(out, A.getData(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(((out[0] == 2)
             && (out[1] == 1)
             && (out[2] == 1)
             && (out[3] == 1)));
}

TEST_CASE("matMul") {
    // C = A @ B
    Matrix<float> A(2, 2);
    Matrix<float> B(2, 2);
    Matrix<float> C(2, 2);
    float in[]  = {2, 1, 1, 1};
    float out[] = {0, 0, 0, 0};
    A.setValues(in);
    B.setValues(in);
    cublasHandle_t handle;
    cublasCreate(&handle);
    matMul(handle, A, B, C);
    cudaMemcpy(out, C.getData(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(((out[0] == 5)
             && (out[1] == 3)
             && (out[2] == 3)
             && (out[3] == 2)));
}

TEST_CASE("matMul_Add") {
    // D = A @ B + C
    Matrix<float> A(2, 2);
    Matrix<float> B(2, 2);
    Matrix<float> C(2, 2);
    Matrix<float> D(2, 2);
    float data0[] = {2, 1, 1, 1};
    float data1[] = {1, 2, 3, 4};
    float out[]   = {0, 0, 0, 0};
    A.setValues(data0);
    B.setValues(data0);
    C.setValues(data1);
    cublasHandle_t handle;
    cublasCreate(&handle);
    matMul_Add(handle, A, B, C, D);
    cudaMemcpy(out, D.getData(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(((out[0] == 6)
             && (out[1] == 5)
             && (out[2] == 6)
             && (out[3] == 6)));
}


TEST_CASE("add") {
    // B = alpha * A + B
    Matrix<float> A(2, 2);
    Matrix<float> B(2, 2);
    float alpha = 0.5;
    float data[] = {1, 2, 3, 4};
    float out[]  = {0, 0, 0, 0};
    A.setValues(data);
    B.setValues(out);
    cublasHandle_t handle;
    cublasCreate(&handle);
    add(handle, A, B, alpha);
    cudaMemcpy(out, B.getData(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(((out[0] == 0.5)
             && (out[1] == 1.0)
             && (out[2] == 1.5)
             && (out[3] == 2.0)));
}


TEST_CASE("matElementMul") {
    // This function applies the Hadamard product
    Matrix<float> A(2, 2);
    Matrix<float> B(2, 2);
    Matrix<float> C(2, 2);
    float data[] = {1, 2, 3, 4};
    float out[]  = {0, 0, 0, 0};
    A.setValues(data);
    B.setValues(data);
    matElementMul(A, B, C);
    cudaMemcpy(out, C.getData(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(((out[0] == 1)
             && (out[1] == 4)
             && (out[2] == 9)
             && (out[3] == 16)));
}


struct addTwo {
    __host__ __device__
    float operator() (float a) const {
        return a + 2.0f;
    }
};

TEST_CASE("matApply") {
    // B = op(A). op is a function that applies elementwise.
    Matrix<float> A(2, 2);
    Matrix<float> B(2, 2);
    float data[] = {1, 2, 3, 4};
    float out[]  = {0, 0, 0, 0};
    A.setValues(data);
    matApply(A, B, addTwo{});
    cudaMemcpy(out, B.getData(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(((out[0] == 3)
             && (out[1] == 4)
             && (out[2] == 5)
             && (out[3] == 6)));
}


TEST_CASE("sparseMatMul") {
    // Like matMul but with csr representation. The Graph class handles
    // this. The values are weird because the graph normalizes the
    // values it parses from the adjacency list values.
    std::vector<std::vector<size_t>> adj_list = {{1}, {0, 2}, {1}};
    cusparseHandle_t sparseHandle;
    cusparseCreate(&sparseHandle);
    Graph<float> g(adj_list, sparseHandle);
    Matrix<float> A(3, 3);
    Matrix<float> B(3, 3);
    float data[] = {1, 1, 0,
                    0, 1, 0,
                    0, 0, 1};
    A.setValues(data);
    sparseMatMul(sparseHandle, g, A, B);
    cudaMemcpy(data, B.getData(), 9 * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(((data[0] == Approx(0.908).epsilon(0.01))
             && (data[1] == Approx(0.741).epsilon(0.01))));
}

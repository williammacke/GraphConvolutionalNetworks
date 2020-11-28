#ifndef GRAPH_H_
#define GRAPH_H_
#include <vector>
#include <cmath>
#include <algorithm>
#include "linAlg/matrix.h"
#include "cusparse.h"

//TODO: Create graph datastructure
//
//

template <class T>
class Graph {
public:
	Graph(std::vector<std::vector<size_t>>& adj_list, cusparseHandle_t handle);
	~Graph();

private:
	T* data;
	int* rowInd;
	int* colInd;
	size_t num_nodes;
	size_t num_edges;
	int nnzb;
	int block_dim;
	int mb;
	cusparseMatDescr_t descr;
public:
	const T* getData() const {
		return data;
	}

	const int* getRowInd() const {
		return rowInd;
	}

	const int* getColInd() const {
		return colInd;
	}

	size_t getNumNodes() const {
		return num_nodes;
	}

	size_t getNumEdges() const {
		return num_edges;
	}

	const cusparseMatDescr_t& getDescr() const {
		return descr;
	}
	int getNNZB() const {
		return nnzb;
	}

	int getBlockDim() const {
		return block_dim;
	}
	int getMB() const {
		return mb;
	}

};


template <class T>
Graph<T>::Graph(std::vector<std::vector<size_t>>& adj_list, cusparseHandle_t handle) {

	auto errs = cusparseCreateMatDescr(&descr);
	if (errs) {
		throw errs;
	} errs = cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
	if (errs) {
		throw errs;
	}

	//descr.MatrixType = CUSPARSE_MATRIX_TYPE_GENERAL;
	//descr.DiagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
	//descr.IndexBase = CUSPARSE_INDEX_BASE_ZERO;

	num_nodes = adj_list.size();
	std::vector<float> degrees(num_nodes);
	num_edges = 0;
	for (int i = 0; i < num_nodes; ++i) {
		std::sort(adj_list[i].begin(), adj_list[i].end());
		num_edges += adj_list[i].size();
		degrees[i] = (adj_list[i].size()+1);
		degrees[i] = pow(degrees[i], -0.5f);
		bool found = false;
		for (int j = 0; j < adj_list[i].size(); ++j) {
			if (adj_list[i][j] == i) {
				found = true;
				break;
			}
		}
		if (!found) {
			num_edges++;
		}
	}
	for (int i = 0; i < num_nodes; ++i) {
		std::cout << i << std::endl;
		for (int j = 0; j < adj_list[i].size(); ++j) {
			std::cout << adj_list[i][j] << " ";
		}
		std::cout << std::endl;
		std::cout << degrees[i] << std::endl;
	}
	std::cin.get();
	float* adj_matrix = new float[num_edges];
	int* rowIndices = new int[num_nodes+1];
	//int* rowIndices = new int[num_edges];
	int* colIndices = new int[num_edges];
	std::cout << num_edges << std::endl;
	int k = 0;
	for (int i = 0; i < num_nodes; ++i) {
		bool found = false;
		rowIndices[i] = k;
		for (int j = 0; j < adj_list[i].size(); ++j) {
			if (adj_list[i][j] == i) {
				found = true;
				adj_matrix[k] = 2*degrees[i]*degrees[adj_list[i][j]];
				colIndices[k] = adj_list[i][j];
				//rowIndices[k] = i;
				++k;
				continue;
			}
			if (adj_list[i][j] > i && !found) {
				found = true;
				adj_matrix[k] = degrees[i]*degrees[i];
				colIndices[k] = i;
				//rowIndices[k] = i;
				++k;
			}
			adj_matrix[k] = degrees[i]*degrees[adj_list[i][j]];
			colIndices[k] = adj_list[i][j];
			//rowIndices[k] = i;
			++k;
		}
		if (! found) {
			adj_matrix[k] = degrees[i]*degrees[i];
			colIndices[k] = i;
			//rowIndices[k] = i;
			++k;
		}
	}
	rowIndices[num_nodes] = num_edges;

	for (int i = 0; i < num_edges; ++i) {
		std::cout << adj_matrix[i] << " ";
	}
	std::cout << std::endl;
	
	auto err = cudaMalloc(&data, sizeof(T)*num_edges);
	if (err) {
		throw err;
	}
	err = cudaMalloc(&rowInd, sizeof(int)*(num_nodes+1));
	if (err) {
		throw err;
	}
	err = cudaMalloc(&colInd, sizeof(int)*num_edges);
	if (err) {
		throw err;
	}

	cudaMemcpy(data, adj_matrix, sizeof(T)*num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(rowInd, rowIndices, sizeof(int)*(num_nodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(colInd, colIndices, sizeof(int)*num_edges, cudaMemcpyHostToDevice);


	//rowIndices[num_nodes] = num_edges;
	std::cout << "k: " << k << " " << num_edges << std::endl;

	delete[] adj_matrix;
	delete[] rowIndices;
	delete[] colIndices;

}

template <class T>
Graph<T>::~Graph() {
	cudaFree(data);
	cudaFree(rowInd);
	cudaFree(colInd);
	cusparseDestroyMatDescr(descr);
}


Matrix<float>& sparseMatMul(cusparseHandle_t handle, const Graph<float>& A, const Matrix<float>& B, Matrix<float>& out, bool transA = false);

#endif

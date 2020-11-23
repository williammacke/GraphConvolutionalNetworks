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
	Graph(std::vector<std::vector<size_t>>& adj_list);
	~Graph();

private:
	T* data;
	int* rowInd;
	int* colInd;
	size_t num_nodes;
	size_t num_edges;
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

};


template <class T>
Graph<T>::Graph(std::vector<std::vector<size_t>>& adj_list) {

	cusparseCreateMatDescr(&descr);
	cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

	//descr.MatrixType = CUSPARSE_MATRIX_TYPE_GENERAL;
	//descr.DiagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
	//descr.IndexBase = CUSPARSE_INDEX_BASE_ZERO;

	num_nodes = adj_list.size();
	std::vector<float> degrees(num_nodes);
	num_edges = 0;
	for (int i = 0; i < num_nodes; ++i) {
		std::sort(adj_list[i].begin(), adj_list[i].end());
		degrees[i] = adj_list[i].size();
		degrees[i] = pow(degrees[i], -0.5f);
		num_edges += adj_list[i].size();
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
	float* adj_matrix = new float[num_edges];
	int* rowIndices = new int[num_nodes+1];
	int* colIndices = new int[num_edges];
	int k = 0;
	for (int i = 0; i < num_nodes; ++i) {
		bool found = false;
		rowIndices[i] = k;
		for (int j = 0; j < adj_list[i].size(); ++j) {
			if (adj_list[i][j] == i) {
				found = true;
				adj_matrix[k] = 2*degrees[i]*degrees[adj_list[i][j]];
				colIndices[k] = adj_list[i][j];
				++k;
				continue;
			}
			if (adj_list[i][j] > i && !found) {
				found = true;
				adj_matrix[k] = degrees[i]*degrees[i];
				colIndices[k] = i;
				++k;
			}
			adj_matrix[k] = degrees[i]*degrees[adj_list[i][j]];
			colIndices[k] = adj_list[i][j];
			++k;
		}
		if (! found) {
			adj_matrix[k] = degrees[i]*degrees[i];
			colIndices[k] = i;
			++k;
		}
	}


	rowIndices[num_nodes] = num_edges;

	cudaMalloc(&data, num_edges*sizeof(T));
	cudaMalloc(&rowInd, (num_nodes+1)*sizeof(int));
	cudaMalloc(&colInd, num_edges*sizeof(int));

	cudaMemcpy(data, adj_matrix, sizeof(T)*num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(rowInd, rowIndices, sizeof(int)*(num_nodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(colInd, colIndices, sizeof(int)*num_edges, cudaMemcpyHostToDevice);

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


Matrix<float>& sparseMatMul(cusparseHandle_t handle, const Graph<float>& A, const Matrix<float>& B, Matrix<float>& out);

#endif

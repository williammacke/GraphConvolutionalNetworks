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
	//int* rowIndices = new int[num_nodes+1];
	int* rowIndices = new int[num_edges];
	int* colIndices = new int[num_edges];
	std::cout << num_edges << std::endl;
	int k = 0;
	for (int i = 0; i < num_nodes; ++i) {
		bool found = false;
		//rowIndices[i] = k;
		for (int j = 0; j < adj_list[i].size(); ++j) {
			if (adj_list[i][j] == i) {
				found = true;
				adj_matrix[k] = 2*degrees[i]*degrees[adj_list[i][j]];
				colIndices[k] = adj_list[i][j];
				rowIndices[k] = i;
				++k;
				continue;
			}
			if (adj_list[i][j] > i && !found) {
				found = true;
				adj_matrix[k] = degrees[i]*degrees[i];
				colIndices[k] = i;
				rowIndices[k] = i;
				++k;
			}
			adj_matrix[k] = degrees[i]*degrees[adj_list[i][j]];
			colIndices[k] = adj_list[i][j];
			rowIndices[k] = i;
			++k;
		}
		if (! found) {
			adj_matrix[k] = degrees[i]*degrees[i];
			colIndices[k] = i;
			rowIndices[k] = i;
			++k;
		}
	}

	for (int i = 0; i < num_edges; ++i) {
		std::cout << adj_matrix[i] << " ";
	}
	std::cout << std::endl;


	//rowIndices[num_nodes] = num_edges;
	std::cout << "k: " << k << " " << num_edges << std::endl;
	std::cin.get();

	float* data_tmp;

	auto err = cudaMalloc(&data_tmp, num_edges*sizeof(T));
	if (err) {
		throw err;
	}
	//err = cudaMalloc(&rowInd, (num_nodes+1)*sizeof(int));
	float* rowInd_tmp2;
	err = cudaMalloc(&rowInd_tmp2, (num_nodes+1)*sizeof(int));
	if (err) {
		throw err;
	}
	int* rowInd_tmp;
	cudaMalloc(&rowInd_tmp, num_edges*sizeof(int));
	float* colInd_tmp;
	err = cudaMalloc(&colInd_tmp, num_edges*sizeof(int));
	if (err) {
		throw err;
	}

	err = cudaMemcpy(data_tmp, adj_matrix, sizeof(T)*num_edges, cudaMemcpyHostToDevice);
	if (err) {
		throw err;
	}
	//err = cudaMemcpy(rowInd, rowIndices, sizeof(int)*(num_nodes+1), cudaMemcpyHostToDevice);
	err = cudaMemcpy(rowInd_tmp, rowIndices, sizeof(int)*num_edges, cudaMemcpyHostToDevice);
	if (err) {
		throw err;
	}
	err = cudaMemcpy(colInd_tmp, colIndices, sizeof(int)*num_edges, cudaMemcpyHostToDevice);
	if (err) {
		throw err;
	}

	cusparseXcoo2csr(handle, rowInd_tmp, num_edges, num_nodes, rowInd_tmp2, CUSPARSE_INDEX_BASE_ZERO);
	auto err2 = cudaDeviceSynchronize();
	if (err2) {
		std::cout << "error: " << err2 << std::endl;
		throw err2;
	}
	cudaFree(rowInd_tmp);


	//int base, nnzb;
	block_dim = std::min(num_nodes, 16);
	int mb = (num_nodes+block_dim-1)/block_dim;

	cudaMalloc(&rowInd, sizeof(int)*(mb+1));

	cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_ROW, num_nodes, num_nodes, descr, rowInd_tmp2, colInd_tmp, block_dim, 
			descr, rowInd, &nnzb);

	cudaMalloc(&colInd, sizeof(int)*nnzb);
	cudamalloc(&data, sizeof(T)*block_dim*block_dim*nnzb);
	cusparseScsr2bsr(handle, CUSPARSE_DIRECTION_ROW, num_nodes, num_nodes, descr, data_tmp,
			rowInd_tmp2, colInd_tmp, block_dim, descr,
			data, rowInd, colInd);
	cudaFree(rowInd_tmp2);
	cudaFree(data_tmp);
	cudaFree(colInd_tmp);

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

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GCN LANGUAGES CXX CUDA)
add_library(gcn STATIC
	linAlg/matrix.h
	linAlg/matrix.cu
	graphes/graph.h
	graphes/graph.cu
	layers/graph_conv.h
	layers/network.h
	functions/activations.h
	functions/activations.cu
	functions/loss.h
	functions/loss.cu
	optimization/gradient_descent.h)

target_compile_features(gcn PUBLIC cxx_std_11)
target_include_directories(gcn PUBLIC ./)
target_link_libraries(gcn PUBLIC cublas cusparse)
set_target_properties( gcn
	                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

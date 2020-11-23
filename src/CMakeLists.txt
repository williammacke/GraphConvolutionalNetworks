cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GCN LANGUAGES CXX CUDA)
add_library(gcn STATIC
	linAlg/matrix.h
	linAlg/matrix.cu)

target_compile_features(gcn PUBLIC cxx_std_11)
target_include_directories(gcn PUBLIC ./)
target_link_libraries(gcn PUBLIC cublas)
set_target_properties( gcn
	                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(graph_cnn LANGUAGES CXX CUDA)

add_executable(test test.cu)
target_link_libraries(test PRIVATE gcn)

add_executable(network network.cu)
target_link_libraries(network PRIVATE gcn)

add_executable(citeseer citeseer.cu)
target_link_libraries(citeseer PRIVATE gcn)

add_executable(citeseer_3layer citeseer_3layer.cu)
target_link_libraries(citeseer_3layer PRIVATE gcn)

add_executable(pubmed pubmed.cu)
target_link_libraries(pubmed PRIVATE gcn)

add_executable(unit unit.cu)
target_link_libraries(unit PRIVATE gcn)

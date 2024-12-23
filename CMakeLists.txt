cmake_minimum_required(VERSION 3.30)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_executable(NeuralNetwork main.cpp
        Neuron.h
        Neuron.cpp
        NeuralNetwork.h
        NeuralNetwork.cpp)

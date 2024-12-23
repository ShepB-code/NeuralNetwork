#pragma once

#ifndef NETWORK_H
#define NETWORK_H

#include "Neuron.h"
#include <vector>

struct SampleData {
    std::vector<float> features;
    int label{};
};

class Layer {
    std::vector<Neuron> neurons;

public:
    Layer(int numNeurons, int numWeights);
    void initNeurons(int numNeurons, int numWeights);
    std::vector<Neuron>& getNeurons() {return neurons;}
};


class NeuralNetwork {
    std::vector<Layer> layers;

public:
    NeuralNetwork(int numInputs, int numHiddenLayers, int numNeuronsInHidden, int numOutputs);
    void addLayer(int numNeurons, int numWeights);
    std::vector<float> forwardPropogate(std::vector<float> inputs);
    void backwardPropogateError(const std::vector<float> &expected);
    void updateWeights(std::vector<float> inputs, float learningRate);
    void train(const std::vector<SampleData>& trainingData, int numEpochs, float learningRate, size_t numOutputs);
    int predict(const std::vector<float>& inputs);
};
#endif //NETWORK_H

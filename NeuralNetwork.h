#pragma once

#ifndef NETWORK_H
#define NETWORK_H

#include "Neuron.h"
#include <vector>

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
};
#endif //NETWORK_H

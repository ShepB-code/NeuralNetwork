#include "Neuron.h"
#include <cstdlib>


Neuron::Neuron(int numWeights) {
    initWeights(numWeights);
}

void Neuron::initWeights(int numWeights) {
    for (int w = 0; w < numWeights; w++) {

        weights.push_back(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
    }
    bias = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

void Neuron::activationFunction(const std::vector<float> &input) {
    activation = bias;
    for (size_t i = 0; i < input.size(); i++) {
        activation += input[i] * weights[i];
    }
}


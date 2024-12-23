#include "Neuron.h"
#include <cstdlib>


Neuron::Neuron(int numWeights) {
    initWeights(numWeights);
}

void Neuron::initWeights(int numWeights) {
    float range = std::sqrt(2.0 / numWeights); // Xavier initialization
    for (int w = 0; w < numWeights; w++) {
        weights.push_back(range * ((static_cast<float>(std::rand()) / RAND_MAX) - 0.5f));
    }
    bias = range * ((static_cast<float>(std::rand()) / RAND_MAX) - 0.5f);
}


void Neuron::activationFunction(const std::vector<float> &input) {
    activation = bias;
    for (size_t i = 0; i < input.size(); i++) {
        activation += input[i] * weights[i];
    }
}


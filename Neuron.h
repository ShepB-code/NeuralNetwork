#pragma once

#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
    std::vector<float> weights;
    float bias{};
    float activation{};
    float transfer{};
    float delta{};

public:
    Neuron(int numWeights);
    void initWeights(int numWeights);
    void activationFunction(const std::vector<float> &input);
    void transferActiviation() {transfer = (1.0f / (1.0f + std::exp(-activation)));}
    float transferDerivative() {return transfer * (1.0f - transfer);}
    float getTransfer() {return transfer;}
    float getDelta() {return delta;}
    void setDelta(float delta) {this->delta = delta;}
    void setBias(float bias) {this->bias = bias;}
    float getBias() {return bias;}
    std::vector<float>& getWeights() {return weights;}
};
#endif //NEURON_H

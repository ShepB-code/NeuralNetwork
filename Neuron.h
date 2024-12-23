#pragma once

#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
    std::vector<float> weights;
    float bias{};
    float activation{};
    float transfer{};

public:
    Neuron(int numWeights);
    void initWeights(int numWeights);
    void activationFunction(const std::vector<float> &input);
    void transferActiviation();
    float getTransfer() {return transfer;}
};
#endif //NEURON_H

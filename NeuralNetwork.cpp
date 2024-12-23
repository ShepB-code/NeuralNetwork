#include "NeuralNetwork.h"
#include <iostream>

/* Layer impl */
Layer::Layer(int numNeurons, int numWeights) {
    initNeurons(numNeurons, numWeights);
}

void Layer::initNeurons(int numNeurons, int numWeights) {
    for (int i = 0; i < numNeurons; i++) neurons.emplace_back(numWeights);
}

/* NeuralNetwork impl */
NeuralNetwork::NeuralNetwork(int numInputs, int numHiddenLayers, int numNeuronsInHidden, int numOutputs) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // hidden layers
    for (int i = 0; i < numHiddenLayers; i++) addLayer(numNeuronsInHidden, numInputs);

    // output layer
    addLayer(numOutputs, numNeuronsInHidden);
}

void NeuralNetwork::addLayer(int numNeurons, int numWeights) {
    layers.emplace_back(numNeurons, numWeights);
}


std::vector<float> NeuralNetwork::forwardPropogate(std::vector<float> inputs) {
    std::vector<float> nextInputs;
    // feed inputs through all layers
    for (size_t i = 0; i < layers.size(); i++) {
        nextInputs.clear();
        // feed inputs through all neurons in layer
        std::vector<Neuron> &layerNeurons = layers[i].getNeurons();
        for (size_t j = 0; j < layerNeurons.size(); j++) {
            layerNeurons[j].activationFunction(inputs);
            layerNeurons[j].transferActiviation();
            nextInputs.emplace_back(layerNeurons[j].getTransfer());
        }
        inputs = nextInputs;
    }
    // represents final output
    return inputs;
}

void NeuralNetwork::backwardPropogateError(const std::vector<float> &expected) {
    for (size_t i = layers.size(); i --> 0;) {

        std::vector<Neuron> &neurons = layers[i].getNeurons();

        // iterate over each neuron in this layer
        for (size_t n = 0; n < neurons.size(); n++)
        {
            float error = 0.0;
            // feed the expected result to the output layer
            if (i == layers.size() - 1)
            {
                error = expected[n] - neurons[n].getTransfer();
            }
            else {
                for (auto& neu : layers[i + 1].getNeurons()) {
                    error += (neu.getWeights()[n] * neu.getDelta());
                }
            }
            // update the delta value of the neuron
            neurons[n].setDelta(error * neurons[n].transferDerivative());
        }

        //
        // // not on output layer
        // if (i != layers.size() - 1) {
        //     for (size_t j = 0; j < neurons.size(); j++) {
        //         float error = 0.0;
        //         for (const auto& weight: neurons[j].getWeights()) {
        //             error += weight * neurons[j].getDelta();
        //         }
        //         errors.emplace_back(error);
        //     }
        // } else {
        //     for (size_t j = 0; j < neurons.size(); j++) {
        //         errors.push_back(neurons[j].getTransfer() - expected[j]);
        //     }
        // }
        // for (size_t j = 0; j < neurons.size(); j++) {
        //     neurons[j].setDelta(errors[j] * neurons[j].transferDerivative());
        // }
    }
}

void NeuralNetwork::updateWeights(std::vector<float> inputs, float learningRate) {
    for (size_t i = 0; i < layers.size(); i++) {
        std::vector<float> nextInput;
        if (i != 0) {
            for (auto & neuron : layers[i - 1].getNeurons()) {
                nextInput.push_back(neuron.getTransfer());
            }
        } else {
            nextInput = inputs;
        }
        std::vector<Neuron> &neurons = layers[i].getNeurons();

        for (auto &neuron : neurons) {
            std::vector<float> &weights = neuron.getWeights();

            // update each weight
            for (size_t j = 0; j < nextInput.size(); j++) {
                weights[j] += learningRate * neuron.getDelta() * nextInput[j];
            }
            // update bias
            neuron.setBias(neuron.getBias() + learningRate * neuron.getDelta());
        }
    }
}

void NeuralNetwork::train(const std::vector<SampleData> &trainingData, int numEpochs, float learningRate, size_t numOutputs) {
    for (int i = 0; i < numEpochs; i++) {
        float sumError = 0.0;
        for (const auto& sample: trainingData) {
            std::vector<float> output = forwardPropogate(sample.features);
            std::vector<float> expected(numOutputs, 0.0);
            expected[sample.label] = 1.0;
            for (size_t j = 0; j < output.size(); j++) {
                sumError += std::pow(output[j] - expected[j], 2);
            }
            backwardPropogateError(expected);
            updateWeights(sample.features, learningRate);
        }
        std::cout << "> Epoch: " << i << " lRate: " << learningRate << " Error: " << sumError << std::endl;
    }
}

int NeuralNetwork::predict(const std::vector<float> &inputs) {
    std::vector<float> output = forwardPropogate(inputs);
    return std::max_element(output.begin(), output.end()) - output.begin();
}







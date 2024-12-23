#include "NeuralNetwork.h"


/* Layer impl */
Layer::Layer(int numNeurons, int numWeights) {
    initNeurons(numNeurons, numWeights);
}

void Layer::initNeurons(int numNeurons, int numWeights) {
    for (int i = 0; i < numNeurons; i++) neurons.emplace_back(numWeights);
}

/* NeuralNetwork impl */
NeuralNetwork::NeuralNetwork(int numInputs, int numHiddenLayers, int numNeuronsInHidden, int numOutputs) {
    // hidden layers
    for (int i = 0; i < numHiddenLayers; i++) addLayer(numNeuronsInHidden, numInputs);

    // output layer
    addLayer(numOutputs, numNeuronsInHidden);
}

void NeuralNetwork::addLayer(int numNeurons, int numWeights) {
    layers.emplace_back(numNeurons, numWeights);
}


std::vector<float> NeuralNetwork::forwardPropogate(std::vector<float> inputs) {
    std::vector<float> newInputs;
    // feed inputs through all layers
    for (size_t i = 0; i < layers.size(); i++) {
        newInputs.clear();
        // feed inputs through all neurons in layer
        std::vector<Neuron> &layerNeurons = layers[i].getNeurons();
        for (size_t j = 0; j < layerNeurons.size(); j++) {
            layerNeurons[j].activationFunction(inputs);
            layerNeurons[j].transferActiviation();
            newInputs.emplace_back(layerNeurons[j].getTransfer());
        }
        inputs = newInputs;
    }
    return inputs;
}



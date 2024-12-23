#include <fstream>
#include <iostream>
#include "NeuralNetwork.h"
#include <string>
#include <sstream>

struct SampleData {
    std::vector<float> features;
    int label{};
};
bool loadDatasetFromCsv(const std::string &filename, std::vector<SampleData> &result);

int main() {
    std::string filename = "WineQT.csv";
    std::vector<SampleData> samples;
    if (!loadDatasetFromCsv(filename, samples)) {
        std::cerr << "Error loading dataset" << std::endl;
    }

    int inputSize = 11;
    int hiddenLayers = 1;
    int neuronsPerHiddenLayer = 11;
    int outputSize = 10;

    NeuralNetwork nn(inputSize, hiddenLayers, neuronsPerHiddenLayer, outputSize);
    std::vector<float> output = nn.forwardPropogate(samples[0].features);

    std::cout << "[";
    for (const auto &out : output) {
        std::cout << out << ",";
    }
    std::cout << "]" << std::endl;
    return 0;
}

bool loadDatasetFromCsv(const std::string &filename, std::vector<SampleData> &result) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open file " << filename << std::endl;
        return false;
    }
    std::string line;
    // skip header
    getline(file, line);

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        SampleData sample;

        // read 11 features TODO: fix this to be dependent on number of features
        for (int i = 0; i < 11; i++) {
            getline(ss, value, ',');
            if (value == "NA") {
                value = "0";
            }
            sample.features.push_back(std::stof(value));
        }
        // get label
        getline(ss, value, ',');
        sample.label = std::stoi(value);
        result.push_back(sample);
    }
    return true;
}
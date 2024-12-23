#include <fstream>
#include <iostream>
#include <random>

#include "NeuralNetwork.h"
#include <string>
#include <sstream>

bool loadDatasetFromCsv(const std::string &filename, std::vector<SampleData> &result);
void splitTrainTest(std::vector<SampleData> &data, std::vector<SampleData> &train, std::vector<SampleData> &test, double trainRatio = 0.8);
int main() {
    std::string filename = "WineQT.csv";
    std::vector<SampleData> samples;
    if (!loadDatasetFromCsv(filename, samples)) {
        std::cerr << "Error loading dataset" << std::endl;
    }

    // split train/test data
    std::vector<SampleData> train, test;
    splitTrainTest(samples, train, test);

    // init neural network
    int inputSize = 11;
    int hiddenLayers = 1;
    int neuronsPerHiddenLayer = 11;
    int outputSize = 11;

    NeuralNetwork nn(inputSize, hiddenLayers, neuronsPerHiddenLayer, outputSize);

    // train neural network
    int numEpochs = 500;
    float learningRate = 0.3;
    nn.train(train, numEpochs, learningRate, outputSize);


    std::vector<int> predictions;
    // make predictions
    for (const auto &sample : test) {
        predictions.emplace_back(nn.predict(sample.features));
    }

    // test accuracy
    int numCorrect = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == test[i].label) {
            numCorrect++;
        }
    }

    std::cout << "Accuracy: " << static_cast<double>(numCorrect) / static_cast<double>(predictions.size()) << std::endl;

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

void splitTrainTest(std::vector<SampleData> &data, std::vector<SampleData> &train, std::vector<SampleData> &test, double trainRatio) {
    auto rng = std::default_random_engine{static_cast<unsigned int>(time(nullptr))};
    std::ranges::shuffle(data, rng);

    auto trainSize = static_cast<size_t>(trainRatio * data.size());

    train = std::vector<SampleData>(data.begin(), data.begin() + trainSize);
    test = std::vector<SampleData>(data.begin() + trainSize, data.end());
}

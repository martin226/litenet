#include "model.h"
#include "matrix.h"
#include "layers.h"
#include "optimizers.h"

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>
#include <ctime>

litenet::Matrix mnistImagesToMatrix(const std::string &path) {
    std::ifstream file (path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(1);
    }
    int magicNumber = 0;
    int numberOfImages = 0;
    int numberOfRows = 0;
    int numberOfCols = 0;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber);
    file.read((char*)&numberOfImages, sizeof(numberOfImages));
    numberOfImages = __builtin_bswap32(numberOfImages);
    file.read((char*)&numberOfRows, sizeof(numberOfRows));
    numberOfRows = __builtin_bswap32(numberOfRows);
    file.read((char*)&numberOfCols, sizeof(numberOfCols));
    numberOfCols = __builtin_bswap32(numberOfCols);
    litenet::Matrix result(numberOfImages, numberOfRows * numberOfCols);
    for (int i = 0; i < numberOfImages; i++) {
        for (int j = 0; j < numberOfRows * numberOfCols; j++) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            result(i, j) = pixel / 255.0;
        }
    }
    return result;
}

litenet::Matrix mnistLabelsToMatrix(const std::string &path) {
    std::ifstream file (path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(1);
    }
    int magicNumber = 0;
    int numberOfItems = 0;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber);
    file.read((char*)&numberOfItems, sizeof(numberOfItems));
    numberOfItems = __builtin_bswap32(numberOfItems);
    litenet::Matrix result(numberOfItems, 10);
    for (int i = 0; i < numberOfItems; i++) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        result(i, label) = 1;
    }
    return result;
}


int main() {
    srand(time(0)); // Seed random number generator

    // Load data
    litenet::Matrix _inputs = mnistImagesToMatrix("data/train-images.idx3-ubyte");
    litenet::Matrix _targets = mnistLabelsToMatrix("data/train-labels.idx1-ubyte");

    // Training set (50000 samples)
    litenet::Matrix trainingInputs = _inputs.subsetRows(0, 49999);
    litenet::Matrix trainingTargets = _targets.subsetRows(0, 49999);

    // Validation set (1000 samples)
    litenet::Matrix validationInputs = _inputs.subsetRows(50000, 59999);
    litenet::Matrix validationTargets = _targets.subsetRows(50000, 59999);

    // Testing set (10000 samples)
    litenet::Matrix testingInputs = mnistImagesToMatrix("data/t10k-images.idx3-ubyte");
    litenet::Matrix testingTargets = mnistLabelsToMatrix("data/t10k-labels.idx1-ubyte");

    std::cout << "Training Inputs shape: " << trainingInputs.getRows() << "x" << trainingInputs.getCols() << std::endl;
    std::cout << "Training Targets shape: " << trainingTargets.getRows() << "x" << trainingTargets.getCols() << std::endl;

    std::cout << "Validation Inputs shape: " << validationInputs.getRows() << "x" << validationInputs.getCols() << std::endl;
    std::cout << "Validation Targets shape: " << validationTargets.getRows() << "x" << validationTargets.getCols() << std::endl;

    std::cout << "Testing Inputs shape: " << testingInputs.getRows() << "x" << testingInputs.getCols() << std::endl;
    std::cout << "Testing Targets shape: " << testingTargets.getRows() << "x" << testingTargets.getCols() << std::endl;

    // Build
    const double learningRate = 0.0001;
    litenet::Model model;
    model.add(std::make_unique<litenet::layers::Dense>(784, 128, "relu", std::make_unique<litenet::initializers::HeUniform>()));    
    model.add(std::make_unique<litenet::layers::Dense>(128, 64, "relu", std::make_unique<litenet::initializers::HeUniform>()));    
    model.add(std::make_unique<litenet::layers::Dense>(64, 32, "relu", std::make_unique<litenet::initializers::HeUniform>()));    
    model.add(std::make_unique<litenet::layers::Dense>(32, 16, "relu", std::make_unique<litenet::initializers::HeUniform>()));    
    model.add(std::make_unique<litenet::layers::Dense>(16, 10, "softmax", std::make_unique<litenet::initializers::GlorotUniform>()));
    model.compile("categorical_crossentropy", std::make_unique<litenet::optimizers::Adam>(learningRate));

    // Train
    const int epochs = 10;
    const int batchSize = 128;
    model.fit(trainingInputs, trainingTargets, epochs, batchSize, validationInputs, validationTargets);

    // Predict
    litenet::Matrix predictions = model.predict(testingInputs);
    // do something with predictions

    // Evaluate
    std::vector<double> results = model.evaluate(testingInputs, testingTargets);
    std::cout << "Loss: " << results[0] << std::endl;
    std::cout << "Accuracy: " << results[1] << std::endl;
    return 0;
}
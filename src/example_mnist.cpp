#include "model.h"
#include "matrix.h"
#include "layers.h"
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

    // Training set
    litenet::Matrix inputs = _inputs.subsetRows(0, 4999);
    litenet::Matrix targets = _targets.subsetRows(0, 4999);

    // Validation set
    litenet::Matrix validationInputs = _inputs.subsetRows(5000, 5099);
    litenet::Matrix validationTargets = _targets.subsetRows(5000, 5099);

    std::cout << "Training Inputs shape: " << inputs.getRows() << "x" << inputs.getCols() << std::endl;
    std::cout << "Training Targets shape: " << targets.getRows() << "x" << targets.getCols() << std::endl;

    std::cout << "Validation Inputs shape: " << validationInputs.getRows() << "x" << validationInputs.getCols() << std::endl;
    std::cout << "Validation Targets shape: " << validationTargets.getRows() << "x" << validationTargets.getCols() << std::endl;

    // print images and labels for the first sample to check if the data is loaded correctly
    for (int j = 0; j < 28 * 28; j++) {
        std::cout << (inputs(0, j) > 0.5 ? "#" : " ");
        if (j % 28 == 27) {
            std::cout << std::endl;
        }
    }
    for (int j = 0; j < 10; j++) {
        if (targets(0, j) == 1) {
            std::cout << j << std::endl;
        }
    }

    // Build
    litenet::Model model;
    model.add(std::make_unique<litenet::layers::Dense>(784, 64, "sigmoid", std::make_unique<litenet::initializers::GlorotUniform>()));
    model.add(std::make_unique<litenet::layers::Dense>(64, 10, "sigmoid", std::make_unique<litenet::initializers::GlorotUniform>()));
    model.compile("mean_squared_error", "sgd", 1);

    // Train
    model.fit(inputs, targets, 30, 64, validationInputs, validationTargets);

    // Predict
    litenet::Matrix predictions = model.predict(validationInputs);

    // for (int i = 0; i < inputs.getRows(); i++) {
    //     for (int j = 0; j < 28 * 28; j++) {
    //         std::cout << (inputs(i, j) > 0.5 ? "#" : " ");
    //         if (j % 28 == 27) {
    //             std::cout << std::endl;
    //         }
    //     }
    //     int label = 0;
    //     for (int j = 0; j < 10; j++) {
    //         if (targets(i, j) == 1) {
    //             label = j;
    //         }
    //     }
    //     std::cout << "Label: " << label << std::endl;
    //     int prediction = 0;
    //     std::cout << "Predictions: ";
    //     for (int j = 0; j < 10; j++) {
    //         std::cout << predictions(i, j) << " ";
    //         if (predictions(i, j) > predictions(i, prediction)) {
    //             prediction = j;
    //         }
    //     }
    //     std::cout << std::endl;
    //     std::cout << "Prediction: " << prediction << std::endl;
    // }

    // Evaluate
    std::vector<double> results = model.evaluate(validationInputs, validationTargets);
    std::cout << "Loss: " << results[0] << std::endl;
    std::cout << "Accuracy: " << results[1] << std::endl;
    return 0;
}
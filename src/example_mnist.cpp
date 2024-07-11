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

    // Train on a subset of the data to speed up the process
    litenet::Matrix inputs = _inputs.subsetRows(0, 10);
    litenet::Matrix targets = _targets.subsetRows(0, 10);

    std::cout << "Inputs shape: " << inputs.getRows() << "x" << inputs.getCols() << std::endl;
    std::cout << "Targets shape: " << targets.getRows() << "x" << targets.getCols() << std::endl;

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

    // Using the model
    litenet::Model model;
    model.add(std::make_unique<litenet::layers::Dense>(784, 32, "relu", std::make_unique<litenet::initializers::HeUniform>()));
    model.add(std::make_unique<litenet::layers::Dense>(32, 10, "softmax", std::make_unique<litenet::initializers::GlorotUniform>()));
    model.compile("categorical_crossentropy", "sgd", 0.01);
    model.fit(inputs, targets, 500, 1);

    // Predictions
    litenet::Matrix predictions = model.predict(inputs);

    for (int i = 0; i < inputs.getRows(); i++) {
        for (int j = 0; j < 28 * 28; j++) {
            std::cout << (inputs(i, j) > 0.5 ? "#" : " ");
            if (j % 28 == 27) {
                std::cout << std::endl;
            }
        }
        int label = 0;
        for (int j = 0; j < 10; j++) {
            if (targets(i, j) == 1) {
                label = j;
            }
        }
        std::cout << "Label: " << label << std::endl;
        int prediction = 0;
        std::cout << "Predictions: ";
        for (int j = 0; j < 10; j++) {
            std::cout << predictions(i, j) << " ";
            if (predictions(i, j) > predictions(i, prediction)) {
                prediction = j;
            }
        }
        std::cout << std::endl;
        std::cout << "Prediction: " << prediction << std::endl;
    }
    return 0;
}
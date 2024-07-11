#include "model.h"
#include "layers.h"
#include "loss.h"
#include <iostream>
#include <random>
#include <memory>
#include <numeric>
#include <algorithm>

litenet::Model::Model() : loss("mean_squared_error"), optimizer("sgd"), learningRate(0.001) {}

void litenet::Model::add(std::unique_ptr<litenet::layers::Layer> layer) {
    layers.push_back(std::move(layer));
}

void litenet::Model::compile(const std::string &loss, const std::string &optimizer, double learningRate) {
    this->loss = loss;
    this->optimizer = optimizer;
    this->learningRate = learningRate;
}

void litenet::Model::fit(const Matrix &inputs, const Matrix &targets, int epochs, int batchSize) {
    // Ensure parameters are valid
    if (layers.empty()) {
        throw std::runtime_error("Model is empty");
    }
    if (inputs.getRows() != targets.getRows()) {
        throw std::invalid_argument("inputs and targets must have the same number of samples");
    }
    if (inputs.getRows() % batchSize != 0) {
        throw std::invalid_argument("batch size must be a divisor of the number of samples");
    }

    // Build the model
    for (const auto &layer : layers) {
        layer->build();
    }

    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
            // TODO: implement batching
            // TODO: implemen shuffling

            Matrix batchInputs = inputs;
            Matrix batchTargets = targets;

            for (const auto &layer : layers) {
                batchInputs = layer->forward(batchInputs);
            }

            Matrix dOutput;
            if (loss == "mean_squared_error") {
                dOutput = litenet::loss::meanSquaredErrorPrime(batchInputs, batchTargets);
            } else if (loss == "mean_absolute_error") {
                dOutput = litenet::loss::meanAbsoluteErrorPrime(batchInputs, batchTargets);
            } else if (loss == "binary_crossentropy") {
                dOutput = litenet::loss::binaryCrossentropyPrime(batchInputs, batchTargets);
            } else if (loss == "categorical_crossentropy") {
                dOutput = litenet::loss::categoricalCrossentropyPrime(batchInputs, batchTargets);
            } else {
                throw std::invalid_argument("unknown loss function");
            }

            for (int j = layers.size() - 1; j >= 0; j--) {
                std::tuple<Matrix, Matrix, Matrix> gradients = layers[j]->backward(dOutput); // dInput, dWeights, dBiases
                dOutput = std::get<0>(gradients);
                Matrix dWeights = std::get<1>(gradients);
                Matrix dBiases = std::get<2>(gradients);

                // TODO: implement optimizers
                if (optimizer == "sgd") {
                    layers[j]->setWeights(layers[j]->getWeights() - learningRate * dWeights);
                    layers[j]->setBiases(layers[j]->getBiases() - learningRate * dBiases);
                } else {
                    throw std::invalid_argument("unknown optimizer");
                }
            }

            // Calculate loss
            double currentLoss;
            if (loss == "mean_squared_error") {
                currentLoss = litenet::loss::meanSquaredError(batchInputs, batchTargets);
            } else if (loss == "mean_absolute_error") {
                currentLoss = litenet::loss::meanAbsoluteError(batchInputs, batchTargets);
            } else if (loss == "binary_crossentropy") {
                currentLoss = litenet::loss::binaryCrossentropy(batchInputs, batchTargets);
            } else if (loss == "categorical_crossentropy") {
                currentLoss = litenet::loss::categoricalCrossentropy(batchInputs, batchTargets);
            } else {
                throw std::invalid_argument("unknown loss function");
            }
            
            // TODO: implement metrics
            std::cout << "Epoch " << (epoch + 1) << " | Loss: " << currentLoss << std::endl;
    }
}

litenet::Matrix litenet::Model::predict(const Matrix &inputs) {
    if (layers.empty()) {
        throw std::runtime_error("Model is empty");
    }
    Matrix predictions = inputs;
    for (const auto &layer : layers) {
        predictions = layer->forward(predictions);
    }
    return predictions;
}
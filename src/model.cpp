#include "model.h"
#include "layers.h"
#include "loss.h"
#include "optimizers.h"

#include <iostream>
#include <random>
#include <memory>
#include <numeric>
#include <algorithm>

namespace litenet {
    Model::Model() : loss("mean_squared_error") {}

    void Model::add(std::unique_ptr<layers::Layer> layer) {
        layers.push_back(std::move(layer));
    }

    void Model::compile(const std::string &loss, std::unique_ptr<optimizers::Optimizer> optimizer) {
        this->loss = loss;
        this->optimizer = std::move(optimizer);
    }

    void Model::fit(const Matrix &inputs, const Matrix &targets, int epochs, int batchSize, const Matrix &validationInputs, const Matrix &validationTargets) {
        // Ensure parameters are valid
        if (layers.empty()) {
            throw std::runtime_error("Model is empty");
        }
        if (inputs.getRows() != targets.getRows()) {
            throw std::invalid_argument("inputs and targets must have the same number of samples");
        }

        // Build the model
        for (const auto &layer : layers) {
            layer->build();
        }

        int numSamples = inputs.getRows();
        int numBatches = numSamples / batchSize;
        if (numSamples % batchSize != 0) {
            numBatches++;
        }

        // Train the model
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle data
            std::vector<int> indices(numSamples);
            std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., numSamples-1
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

            // Create shuffled inputs and targets
            Matrix shuffledInputs(inputs.getRows(), inputs.getCols());
            Matrix shuffledTargets(targets.getRows(), targets.getCols());
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < inputs.getCols(); j++) {
                    shuffledInputs(i, j) = inputs(indices[i], j);
                }
                for (int j = 0; j < targets.getCols(); j++) {
                    shuffledTargets(i, j) = targets(indices[i], j);
                }
            }

            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                int startIdx = batchIndex * batchSize;
                int endIdx = startIdx + batchSize;
                if (endIdx > numSamples) { // Last batch
                    endIdx = numSamples;
                }

                // Create batch inputs and targets
                Matrix batchInputs(batchSize, inputs.getCols());
                Matrix batchTargets(batchSize, targets.getCols());
                for (int i = startIdx; i < endIdx; i++) {
                    for (int j = 0; j < inputs.getCols(); j++) {
                        batchInputs(i - startIdx, j) = shuffledInputs(i, j);
                    }
                    for (int j = 0; j < targets.getCols(); j++) {
                        batchTargets(i - startIdx, j) = shuffledTargets(i, j);
                    }
                }

                // Forward pass
                Matrix predictions = batchInputs;
                for (const auto &layer : layers) {
                    predictions = layer->forward(predictions);
                }

                // Compute loss and its derivative
                Matrix dOutput;
                if (loss == "mean_squared_error") {
                    dOutput = litenet::loss::meanSquaredErrorPrime(predictions, batchTargets);
                } else if (loss == "mean_absolute_error") {
                    dOutput = litenet::loss::meanAbsoluteErrorPrime(predictions, batchTargets);
                } else if (loss == "binary_crossentropy") {
                    dOutput = litenet::loss::binaryCrossentropyPrime(predictions, batchTargets);
                } else if (loss == "categorical_crossentropy") {
                    dOutput = litenet::loss::categoricalCrossentropyPrime(predictions, batchTargets);
                } else {
                    throw std::invalid_argument("unknown loss function");
                }

                // Backward pass and weight updates
                for (int j = layers.size() - 1; j >= 0; j--) {
                    dOutput = layers[j]->backward(dOutput);

                    // Update weights and biases
                    optimizer->update(*layers[j]);
                }
            }

            // Calculate loss over entire dataset for reporting
            Matrix predictions = inputs;
            for (const auto &layer : layers) {
                predictions = layer->forward(predictions);
            }

            double currentLoss;
            if (loss == "mean_squared_error") {
                currentLoss = litenet::loss::meanSquaredError(predictions, targets);
            } else if (loss == "mean_absolute_error") {
                currentLoss = litenet::loss::meanAbsoluteError(predictions, targets);
            } else if (loss == "binary_crossentropy") {
                currentLoss = litenet::loss::binaryCrossentropy(predictions, targets);
            } else if (loss == "categorical_crossentropy") {
                currentLoss = litenet::loss::categoricalCrossentropy(predictions, targets);
            } else {
                throw std::invalid_argument("unknown loss function");
            }
        
            if (validationInputs.getRows() == 0) { // No validation set
                // Print epoch loss
                std::cout << "Epoch " << (epoch + 1) << " | loss: " << currentLoss << std::endl;
                continue;
            }

            // Calculate validation loss
            Matrix validationPredictions = validationInputs;
            for (const auto &layer : layers) {
                validationPredictions = layer->forward(validationPredictions);
            }

            double validationLoss;
            if (loss == "mean_squared_error") {
                validationLoss = litenet::loss::meanSquaredError(validationPredictions, validationTargets);
            } else if (loss == "mean_absolute_error") {
                validationLoss = litenet::loss::meanAbsoluteError(validationPredictions, validationTargets);
            } else if (loss == "binary_crossentropy") {
                validationLoss = litenet::loss::binaryCrossentropy(validationPredictions, validationTargets);
            } else if (loss == "categorical_crossentropy") {
                validationLoss = litenet::loss::categoricalCrossentropy(validationPredictions, validationTargets);
            } else {
                throw std::invalid_argument("unknown loss function");
            }

            // Print epoch loss
            std::cout << "Epoch " << (epoch + 1) << " | loss: " << currentLoss << " | val_loss: " << validationLoss << std::endl;
        }
    }

    Matrix Model::predict(const Matrix &inputs) {
        if (layers.empty()) {
            throw std::runtime_error("Model is empty");
        }
        Matrix predictions = inputs;
        for (const auto &layer : layers) {
            predictions = layer->forward(predictions);
        }
        return predictions;
    }

    std::vector<double> Model::evaluate(const Matrix &inputs, const Matrix &targets) {
        if (layers.empty()) {
            throw std::runtime_error("Model is empty");
        }
        Matrix predictions = predict(inputs);
        std::vector<double> results;
        if (loss == "mean_squared_error") {
            results.push_back(litenet::loss::meanSquaredError(predictions, targets));
        } else if (loss == "mean_absolute_error") {
            results.push_back(litenet::loss::meanAbsoluteError(predictions, targets));
        } else if (loss == "binary_crossentropy") {
            results.push_back(litenet::loss::binaryCrossentropy(predictions, targets));
        } else if (loss == "categorical_crossentropy") {
            results.push_back(litenet::loss::categoricalCrossentropy(predictions, targets));
        } else {
            throw std::invalid_argument("unknown loss function");
        }

        // Accuracy
        int correct = 0;
        for (int i = 0; i < predictions.getRows(); i++) {
            int targetIndex = -1;
            int predictionIndex = -1;
            for (int j = 0; j < predictions.getCols(); j++) {
                if (targets(i, j) == 1) {
                    targetIndex = j;
                }
                if (predictions(i, j) > predictions(i, predictionIndex)) {
                    predictionIndex = j;
                }
            }
            if (targetIndex == predictionIndex) {
                correct++;
            }
        }
        results.push_back(static_cast<double>(correct) / predictions.getRows());

        return results;
    }
}
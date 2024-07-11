#include "layers.h"
#include "activations.h"
#include "loss.h"
#include "initializers.h"

#include <stdexcept>

namespace litenet::layers {
    Matrix Layer::getWeights() const {
        return weights;
    }

    Matrix Layer::getBiases() const {
        return biases;
    }

    void Layer::setWeights(const Matrix &weights) {
        this->weights = weights;
    }

    void Layer::setBiases(const Matrix &biases) {
        this->biases = biases;
    }

    Dense::Dense(int inFeatures, int outFeatures, const std::string &activation, std::unique_ptr<initializers::Initializer> kernel_initializer, std::unique_ptr<initializers::Initializer> bias_initializer) {
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->activation = activation;
        this->kernel_initializer = std::move(kernel_initializer);
        this->bias_initializer = std::move(bias_initializer);
    }

    void Dense::build() {
        // for now, inputs is a matrix of shape (samples, features)
        // weights is a matrix of shape (features, units)
        // biases is a matrix of shape (units,)
        weights = kernel_initializer->initialize(inFeatures, outFeatures);
        biases = bias_initializer->initialize(outFeatures, 1);
    }

    Matrix Dense::forward(const Matrix &inputs) {
        // matrix multiplication:
        // inputs: (samples, features)
        // weights: (features, units)
        // inputs * weights: (samples, units)
        // biases: (units,)
        // 
        // z = inputs * weights + biases
        this->inputs = inputs;
        Matrix z = inputs * weights;

        for (int i = 0; i < z.getRows(); i++) {
            for (int j = 0; j < z.getCols(); j++) {
                z(i, j) += biases(j, 0);
            }
        }

        return applyActivation(z);
    }

    std::tuple<Matrix, Matrix, Matrix> Dense::backward(const Matrix &dOutput) {
        // Compute pre-activation
        Matrix z = inputs * weights;

        // Add biases
        for (int i = 0; i < z.getRows(); i++) {
            for (int j = 0; j < z.getCols(); j++) {
                z(i, j) += biases(j, 0);
            }
        }

        // Compute derivative of the activation function with respect to the pre-activation
        Matrix dActivation = applyActivationPrime(z);
        
        // Compute delta as the Hadamard product of dOutput and dActivation
        Matrix delta = dOutput.hadamard(dActivation);

        // Compute gradients with respect to the weights and biases
        Matrix dWeights = inputs.transpose() * delta;
        Matrix dBiases = delta.sum(0).transpose(); // column-wise sum and then transpose to match the shape of biases

        // Compute gradient with respect to the input
        Matrix dInputs = delta * weights.transpose();

        // Return dInputs, dWeights, and dBiases
        return std::make_tuple(dInputs, dWeights, dBiases);
    }

    Matrix Dense::applyActivation(const Matrix &m) {
        if (activation == "sigmoid") {
            return activations::sigmoid(m);
        } else if (activation == "relu") {
            return activations::relu(m);
        } else if (activation == "leakyRelu") {
            return activations::leakyRelu(m);
        } else if (activation == "tanh") {
            return activations::tanh(m);
        } else if (activation == "softmax") {
            return activations::softmax(m);
        }
        return m;
    }

    Matrix Dense::applyActivationPrime(const Matrix &m) {
        if (activation == "sigmoid") {
            return activations::sigmoidPrime(m);
        } else if (activation == "relu") {
            return activations::reluPrime(m);
        } else if (activation == "leakyRelu") {
            return activations::leakyReluPrime(m);
        } else if (activation == "tanh") {
            return activations::tanhPrime(m);
        } else if (activation == "softmax") {
            return activations::softmaxPrime(m);
        }
        return m;
    }
}
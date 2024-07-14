#include "layers.h"
#include "activations.h"
#include "loss.h"
#include "initializers.h"

#include <stdexcept>

namespace litenet::layers {
    std::string Layer::getName() const {
        return name;
    }
    int Layer::getInFeatures() const {
        return inFeatures;
    }
    int Layer::getOutFeatures() const {
        return outFeatures;
    }
    int Layer::getNumParameters() const {
        return inFeatures * outFeatures + outFeatures; // change later
    }
    Dense::Dense(int inFeatures, int outFeatures, const std::string &activation, std::unique_ptr<initializers::Initializer> kernel_initializer, std::unique_ptr<initializers::Initializer> bias_initializer) {
        this->name = "Dense";
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
        this->parameters["weights"] = kernel_initializer->initialize(inFeatures, outFeatures);
        this->parameters["biases"] = bias_initializer->initialize(outFeatures, 1);
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
        Matrix z = inputs * this->parameters["weights"];

        for (int i = 0; i < z.getRows(); i++) {
            for (int j = 0; j < z.getCols(); j++) {
                z(i, j) += this->parameters["biases"](j, 0);
            }
        }

        return applyActivation(z);
    }

    Matrix Dense::backward(const Matrix &dOutput) {
        // Compute pre-activation
        Matrix z = inputs * this->parameters["weights"];

        // Add biases
        for (int i = 0; i < z.getRows(); i++) {
            for (int j = 0; j < z.getCols(); j++) {
                z(i, j) += this->parameters["biases"](j, 0);
            }
        }

        // Compute derivative of the activation function with respect to the pre-activation
        Matrix dActivation = applyActivationPrime(z);
        
        // Compute delta as the Hadamard product of dOutput and dActivation
        Matrix delta = dOutput.hadamard(dActivation);

        // Compute gradients with respect to the weights and biases
        this->gradients["weights"] = inputs.transpose() * delta;
        this->gradients["biases"] = delta.sum(0).transpose(); // column-wise sum and then transpose to match the shape of biases

        // Compute gradient with respect to the input
        Matrix dInputs = delta * this->parameters["weights"].transpose();

        return dInputs;
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

    Dropout::Dropout(float rate) {
        this->name = "Dropout";
        this->rate = rate;
    }

    void Dropout::build() {
        // Nothing to do here
    }

    Matrix Dropout::forward(const Matrix &inputs) {
        // Generate a mask with the same shape as the inputs
        mask = randomUniform.initialize(inputs.getRows(), inputs.getCols());
        for (int i = 0; i < mask.getRows(); i++) {
            for (int j = 0; j < mask.getCols(); j++) {
                mask(i, j) = mask(i, j) > rate ? 1 : 0;
            }
        }
        return inputs.hadamard(mask);
    }

    Matrix Dropout::backward(const Matrix &dOutput) {
        return dOutput.hadamard(mask);
    }
}
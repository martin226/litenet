#ifndef LAYERS_H
#define LAYERS_H

#include "matrix.h"
#include "activations.h"
#include "initializers.h"

#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include <unordered_map>

namespace litenet::layers {
    class Layer {
        public:
            Layer() {}
            virtual void build() = 0;
            virtual Matrix forward(const Matrix &inputs) = 0;
            virtual Matrix backward(const Matrix &dOutput) = 0;
            std::string getName() const;
            int getInFeatures() const;
            int getOutFeatures() const;
            int getNumParameters() const;
            std::unordered_map<std::string, Matrix> parameters;
            std::unordered_map<std::string, Matrix> gradients;
        protected:
            std::string name;
            int inFeatures;
            int outFeatures;
    };
    class Dense : public Layer {
        public:
            Dense(int inFeatures, int outFeatures, const std::string &activation = "linear", std::unique_ptr<initializers::Initializer> kernel_initializer = std::make_unique<initializers::GlorotUniform>(), std::unique_ptr<initializers::Initializer> bias_initializer = std::make_unique<initializers::Zeros>());
            void build() override;
            Matrix forward(const Matrix &inputs) override;
            Matrix backward(const Matrix &dOutput) override;    
        private:
            std::unique_ptr<initializers::Initializer> kernel_initializer;
            std::unique_ptr<initializers::Initializer> bias_initializer;
            std::string activation;
            Matrix inputs;
            Matrix applyActivation(const Matrix &m);
            Matrix applyActivationPrime(const Matrix &m);
    };
    class Dropout : public Layer {
        public:
            Dropout(float rate = 0.5);
            void build() override;
            Matrix forward(const Matrix &inputs) override;
            Matrix backward(const Matrix &dOutput) override;
        private:
            float rate;
            Matrix mask;
            initializers::RandomUniform randomUniform = initializers::RandomUniform(0, 1);
    };
}

#endif
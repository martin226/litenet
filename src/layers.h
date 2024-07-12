#ifndef LAYERS_H
#define LAYERS_H

#include "matrix.h"
#include "activations.h"
#include "initializers.h"

#include <vector>
#include <tuple>
#include <string>
#include <memory>

namespace litenet::layers {
    class Layer {
        public:
            Layer() {}
            virtual void build() = 0;
            virtual Matrix forward(const Matrix &inputs) = 0;
            virtual std::tuple<Matrix, Matrix, Matrix> backward(const Matrix &dOutput) = 0;
            Matrix getWeights() const;
            Matrix getBiases() const;
            void setWeights(const Matrix &weights);
            void setBiases(const Matrix &biases);
            void updateWeights(const Matrix &dWeights);
            void updateBiases(const Matrix &dBiases);
        protected:
            Matrix weights;
            Matrix biases;
            virtual Matrix applyActivation(const Matrix &m) = 0;
            virtual Matrix applyActivationPrime(const Matrix &m) = 0;
    };
    class Dense : public Layer {
        public:
            Dense(int inFeatures, int outFeatures, const std::string &activation = "linear", std::unique_ptr<initializers::Initializer> kernel_initializer = std::make_unique<initializers::GlorotUniform>(), std::unique_ptr<initializers::Initializer> bias_initializer = std::make_unique<initializers::Zeros>());
            void build() override;
            Matrix forward(const Matrix &inputs) override;
            std::tuple<Matrix, Matrix, Matrix> backward(const Matrix &dOutput) override;    
        private:
            std::unique_ptr<initializers::Initializer> kernel_initializer;
            std::unique_ptr<initializers::Initializer> bias_initializer;
            int inFeatures;
            int outFeatures;
            std::string activation;
            Matrix inputs;
            Matrix applyActivation(const Matrix &m) override;
            Matrix applyActivationPrime(const Matrix &m) override;
    };
}

#endif
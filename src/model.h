#ifndef MODEL_H
#define MODEL_H

#include "layers.h"
#include <vector>
#include <memory>

namespace litenet {
    class Model {
        public:
            Model();
            void add(std::unique_ptr<layers::Layer> layer);
            void compile(const std::string &loss, const std::string &optimizer, double learningRate = 0.001);
            void fit(const Matrix &inputs, const Matrix &targets, int epochs, int batchSize = 32);
            Matrix predict(const Matrix &inputs);
        private:
            std::vector<std::unique_ptr<layers::Layer>> layers;
            std::string loss;
            std::string optimizer;
            double learningRate;
    };
}

#endif
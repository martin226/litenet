#ifndef MODEL_H
#define MODEL_H

#include "layers.h"
#include "optimizers.h"

#include <vector>
#include <memory>

namespace litenet {
    class Model {
        public:
            Model();
            void add(std::unique_ptr<layers::Layer> layer);
            void compile(const std::string &loss, const std::unique_ptr<optimizers::Optimizer> optimizer);
            void fit(const Matrix &inputs, const Matrix &targets, int epochs, int batchSize = 32, const Matrix &validationInputs = Matrix(), const Matrix &validationTargets = Matrix());
            Matrix predict(const Matrix &inputs);
            std::vector<double> evaluate(const Matrix &inputs, const Matrix &targets);
        private:
            std::vector<std::unique_ptr<layers::Layer>> layers;
            std::string loss;
            std::unique_ptr<optimizers::Optimizer> optimizer;
    };
}

#endif
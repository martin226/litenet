#include "optimizers.h"
#include "layers.h"

#include <cmath>

namespace litenet::optimizers {
    Optimizer::Optimizer(double learningRate) : learningRate(learningRate) {}

    SGD::SGD(double learningRate) : Optimizer(learningRate) {}

    void SGD::update(layers::Layer &layer) {
        for (auto it = layer.parameters.begin(); it != layer.parameters.end(); it++) {
            std::string name = it->first;
            Matrix &parameter = it->second;
            Matrix &dParameter = layer.gradients[name];
            parameter -= dParameter * learningRate;
        }
    }

    Adam::Adam(double learningRate, double beta1, double beta2, double epsilon) : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void Adam::update(layers::Layer &layer) {
        t++;

        for (auto it = layer.parameters.begin(); it != layer.parameters.end(); it++) {
            std::string name = it->first;
            Matrix &parameter = it->second;
            Matrix &dParameter = layer.gradients[name];

            int rows = parameter.getRows();
            int cols = parameter.getCols();

            if (m.find(name) == m.end()) {
                m[name] = Matrix(rows, cols);
            }

            if (v.find(name) == v.end()) {
                v[name] = Matrix(rows, cols);
            }

            if (m[name].getRows() != rows || m[name].getCols() != cols) {
                m[name] = Matrix(rows, cols);
            }

            if (v[name].getRows() != rows || v[name].getCols() != cols) {
                v[name] = Matrix(rows, cols);
            }

            m[name] = beta1 * m[name] + (1 - beta1) * dParameter;
            v[name] = beta2 * v[name] + (1 - beta2) * dParameter.pow(2);

            Matrix mHat = m[name] / (1 - std::pow(beta1, t));
            Matrix vHat = v[name] / (1 - std::pow(beta2, t));

            parameter -= mHat / (vHat.sqrt() + epsilon) * learningRate;
        }
    }

    AdamW::AdamW(double learningRate, double weightDecay, double beta1, double beta2, double epsilon) : Optimizer(learningRate), weightDecay(weightDecay), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void AdamW::update(layers::Layer &layer) {
        t++;

        for (auto it = layer.parameters.begin(); it != layer.parameters.end(); it++) {
            std::string name = it->first;
            Matrix &parameter = it->second;
            Matrix &dParameter = layer.gradients[name];

            int rows = parameter.getRows();
            int cols = parameter.getCols();

            if (m.find(name) == m.end()) {
                m[name] = Matrix(rows, cols);
            }

            if (v.find(name) == v.end()) {
                v[name] = Matrix(rows, cols);
            }

            if (m[name].getRows() != rows || m[name].getCols() != cols) {
                m[name] = Matrix(rows, cols);
            }

            if (v[name].getRows() != rows || v[name].getCols() != cols) {
                v[name] = Matrix(rows, cols);
            }

            m[name] = beta1 * m[name] + (1 - beta1) * dParameter;
            v[name] = beta2 * v[name] + (1 - beta2) * dParameter.pow(2);

            Matrix mHat = m[name] / (1 - std::pow(beta1, t));
            Matrix vHat = v[name] / (1 - std::pow(beta2, t));

            parameter -= mHat / (vHat.sqrt() + epsilon) * learningRate;

            if (weightDecay > 0 && name == "weights") {
                parameter -= parameter * weightDecay * learningRate;
            }
        }
    }

    AdaGrad::AdaGrad(double learningRate, double epsilon) : Optimizer(learningRate), epsilon(epsilon) {}

    void AdaGrad::update(layers::Layer &layer) {
        for (auto it = layer.parameters.begin(); it != layer.parameters.end(); it++) {
            std::string name = it->first;
            Matrix &parameter = it->second;
            Matrix &dParameter = layer.gradients[name];
            
            int rows = parameter.getRows();
            int cols = parameter.getCols();

            if (v.find(name) == v.end()) {
                v[name] = Matrix(rows, cols);
            }

            if (v[name].getRows() != rows || v[name].getCols() != cols) {
                v[name] = Matrix(rows, cols);
            }

            v[name] += dParameter.pow(2);

            parameter -= dParameter / (v[name].sqrt() + epsilon) * learningRate;
        }
    }

    RMSProp::RMSProp(double learningRate, double beta, double epsilon) : Optimizer(learningRate), beta(beta), epsilon(epsilon) {}

    void RMSProp::update(layers::Layer &layer) {
        for (auto it = layer.parameters.begin(); it != layer.parameters.end(); it++) {
            std::string name = it->first;
            Matrix &parameter = it->second;
            Matrix &dParameter = layer.gradients[name];
            
            int rows = parameter.getRows();
            int cols = parameter.getCols();

            if (v.find(name) == v.end()) {
                v[name] = Matrix(rows, cols);
            }

            if (v[name].getRows() != rows || v[name].getCols() != cols) {
                v[name] = Matrix(rows, cols);
            }

            v[name] = beta * v[name] + (1 - beta) * dParameter.pow(2);

            parameter -= dParameter / (v[name].sqrt() + epsilon) * learningRate;
        }
    }
}
#include "optimizers.h"

#include <cmath>

namespace litenet::optimizers {
    Optimizer::Optimizer(double learningRate) : learningRate(learningRate) {}

    SGD::SGD(double learningRate) : Optimizer(learningRate) {}

    void SGD::update(Matrix &weights, Matrix &biases, const Matrix &dWeights, const Matrix &dBiases) {
        weights -= dWeights * learningRate;
        biases -= dBiases * learningRate;
    }

    Adam::Adam(double learningRate, double beta1, double beta2, double epsilon) : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void Adam::update(Matrix &weights, Matrix &biases, const Matrix &dWeights, const Matrix &dBiases) {
        t++;
        if (mWeights.empty()) {
            mWeights.push_back(Matrix(weights.getRows(), weights.getCols()));
            vWeights.push_back(Matrix(weights.getRows(), weights.getCols()));
            mBiases.push_back(Matrix(biases.getRows(), biases.getCols()));
            vBiases.push_back(Matrix(biases.getRows(), biases.getCols()));
        }
        if (mWeights[0].getRows() != weights.getRows() || mWeights[0].getCols() != weights.getCols()) {
            mWeights[0] = Matrix(weights.getRows(), weights.getCols());
            vWeights[0] = Matrix(weights.getRows(), weights.getCols());
            mBiases[0] = Matrix(biases.getRows(), biases.getCols());
            vBiases[0] = Matrix(biases.getRows(), biases.getCols());
        }
        mWeights[0] = beta1 * mWeights[0] + (1 - beta1) * dWeights;
        vWeights[0] = beta2 * vWeights[0] + (1 - beta2) * dWeights.pow(2);
        mBiases[0] = beta1 * mBiases[0] + (1 - beta1) * dBiases;
        vBiases[0] = beta2 * vBiases[0] + (1 - beta2) * dBiases.pow(2);
        Matrix mWeightsCorrected = mWeights[0] / (1 - std::pow(beta1, t));
        Matrix vWeightsCorrected = vWeights[0] / (1 - std::pow(beta2, t));
        Matrix mBiasesCorrected = mBiases[0] / (1 - std::pow(beta1, t));
        Matrix vBiasesCorrected = vBiases[0] / (1 - std::pow(beta2, t));

        weights -= mWeightsCorrected / (vWeightsCorrected.sqrt() + epsilon) * learningRate;
        biases -= mBiasesCorrected / (vBiasesCorrected.sqrt() + epsilon) * learningRate;
    }

    AdaGrad::AdaGrad(double learningRate, double epsilon) : Optimizer(learningRate), epsilon(epsilon), vWeights(0, 0), vBiases(0, 0) {}

    void AdaGrad::update(Matrix &weights, Matrix &biases, const Matrix &dWeights, const Matrix &dBiases) {
        if (vWeights.getRows() != weights.getRows() || vWeights.getCols() != weights.getCols()) {
            vWeights = Matrix(weights.getRows(), weights.getCols());
            vBiases = Matrix(biases.getRows(), biases.getCols());
        }
        vWeights += dWeights.pow(2);
        vBiases += dBiases.pow(2);

        weights -= dWeights / (vWeights.sqrt() + epsilon) * learningRate;
        biases -= dBiases / (vBiases.sqrt() + epsilon) * learningRate;
    }
}
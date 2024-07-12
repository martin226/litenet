#include "optimizers.h"
#include "layers.h"

#include <cmath>

namespace litenet::optimizers {
    Optimizer::Optimizer(double learningRate) : learningRate(learningRate) {}

    SGD::SGD(double learningRate) : Optimizer(learningRate) {}

    void SGD::update(layers::Layer &layer, const Matrix &dWeights, const Matrix &dBiases) {
        layer.updateWeights(dWeights * learningRate);
        layer.updateBiases(dBiases * learningRate);
    }

    Adam::Adam(double learningRate, double beta1, double beta2, double epsilon) : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void Adam::update(layers::Layer &layer, const Matrix &dWeights, const Matrix &dBiases) {
        int weightsRows = layer.getWeights().getRows();
        int weightsCols = layer.getWeights().getCols();
        int biasesRows = layer.getBiases().getRows();
        int biasesCols = layer.getBiases().getCols();

        t++;
        if (mWeights.empty()) {
            mWeights.push_back(Matrix(weightsRows, weightsCols));
            vWeights.push_back(Matrix(weightsRows, weightsCols));
            mBiases.push_back(Matrix(biasesRows, biasesCols));
            vBiases.push_back(Matrix(biasesRows, biasesCols));
        }
        if (mWeights[0].getRows() != weightsRows || mWeights[0].getCols() != weightsCols) {
            mWeights[0] = Matrix(weightsRows, weightsCols);
            vWeights[0] = Matrix(weightsRows, weightsCols);
            mBiases[0] = Matrix(biasesRows, biasesCols);
            vBiases[0] = Matrix(biasesRows, biasesCols);
        }
        mWeights[0] = beta1 * mWeights[0] + (1 - beta1) * dWeights;
        vWeights[0] = beta2 * vWeights[0] + (1 - beta2) * dWeights.pow(2);
        mBiases[0] = beta1 * mBiases[0] + (1 - beta1) * dBiases;
        vBiases[0] = beta2 * vBiases[0] + (1 - beta2) * dBiases.pow(2);
        Matrix mWeightsCorrected = mWeights[0] / (1 - std::pow(beta1, t));
        Matrix vWeightsCorrected = vWeights[0] / (1 - std::pow(beta2, t));
        Matrix mBiasesCorrected = mBiases[0] / (1 - std::pow(beta1, t));
        Matrix vBiasesCorrected = vBiases[0] / (1 - std::pow(beta2, t));

        layer.updateWeights(mWeightsCorrected / (vWeightsCorrected.sqrt() + epsilon) * learningRate);
        layer.updateBiases(mBiasesCorrected / (vBiasesCorrected.sqrt() + epsilon) * learningRate);
    }

    AdaGrad::AdaGrad(double learningRate, double epsilon) : Optimizer(learningRate), epsilon(epsilon), vWeights(0, 0), vBiases(0, 0) {}

    void AdaGrad::update(layers::Layer &layer, const Matrix &dWeights, const Matrix &dBiases) {
        int weightsRows = layer.getWeights().getRows();
        int weightsCols = layer.getWeights().getCols();
        int biasesRows = layer.getBiases().getRows();
        int biasesCols = layer.getBiases().getCols();

        if (vWeights.getRows() != weightsRows || vWeights.getCols() != weightsCols) {
            vWeights = Matrix(weightsRows, weightsCols);
            vBiases = Matrix(biasesRows, biasesCols);
        }
        vWeights += dWeights.pow(2);
        vBiases += dBiases.pow(2);

        layer.updateWeights(dWeights / (vWeights.sqrt() + epsilon) * learningRate);
        layer.updateBiases(dBiases / (vBiases.sqrt() + epsilon) * learningRate);
    }

    RMSProp::RMSProp(double learningRate, double beta, double epsilon) : Optimizer(learningRate), beta(beta), epsilon(epsilon), vWeights(0, 0), vBiases(0, 0) {}

    void RMSProp::update(layers::Layer &layer, const Matrix &dWeights, const Matrix &dBiases) {
        int weightsRows = layer.getWeights().getRows();
        int weightsCols = layer.getWeights().getCols();
        int biasesRows = layer.getBiases().getRows();
        int biasesCols = layer.getBiases().getCols();

        if (vWeights.getRows() != weightsRows || vWeights.getCols() != weightsCols) {
            vWeights = Matrix(weightsRows, weightsCols);
            vBiases = Matrix(biasesRows, biasesCols);
        }
        vWeights = beta * vWeights + (1 - beta) * dWeights.pow(2);
        vBiases = beta * vBiases + (1 - beta) * dBiases.pow(2);

        layer.updateWeights(dWeights / (vWeights.sqrt() + epsilon) * learningRate);
        layer.updateBiases(dBiases / (vBiases.sqrt() + epsilon) * learningRate);
    }
}
#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "matrix.h"

#include <vector>

namespace litenet::optimizers {
    class Optimizer {
        public:
            Optimizer(double learningRate);
            virtual void update(Matrix &weights, Matrix &biases, const Matrix &dWeights, const Matrix &dBiases) = 0;
        protected:
            double learningRate;
    };
    class SGD : public Optimizer {
        public:
            SGD(double learningRate = 0.1);
            void update(Matrix &weights, Matrix &biases, const Matrix &dWeights, const Matrix &dBiases) override;
    };
    class Adam : public Optimizer {
        public:
            Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
            void update(Matrix &weights, Matrix &biases, const Matrix &dWeights, const Matrix &dBiases) override;
        private:
            double beta1;
            double beta2;
            double epsilon;
            std::vector<Matrix> mWeights;
            std::vector<Matrix> vWeights;
            std::vector<Matrix> mBiases;
            std::vector<Matrix> vBiases;
            int t;
    };
    class AdaGrad : public Optimizer {
        public:
            AdaGrad(double learningRate = 0.01, double epsilon = 1e-8);
            void update(Matrix &weights, Matrix &biases, const Matrix &dWeights, const Matrix &dBiases) override;
        private:
            double epsilon;
            Matrix vWeights;
            Matrix vBiases;
    };
    class RMSProp : public Optimizer {
        public:
            RMSProp(double learningRate = 0.01, double beta = 0.9, double epsilon = 1e-8);
            void update(Matrix &weights, Matrix &biases, const Matrix &dWeights, const Matrix &dBiases) override;
        private:
            double beta;
            double epsilon;
            Matrix vWeights;
            Matrix vBiases;
    };
}

#endif
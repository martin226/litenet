#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "matrix.h"
#include "layers.h"

#include <vector>
#include <unordered_map>

namespace litenet::optimizers {
    class Optimizer {
        public:
            Optimizer(double learningRate);
            virtual ~Optimizer() {}
            virtual void update(layers::Layer &layer) = 0;
        protected:
            double learningRate;
    };
    class SGD : public Optimizer {
        public:
            SGD(double learningRate = 0.1);
            void update(layers::Layer &layer) override;
    };
    class Adam : public Optimizer {
        public:
            Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
            void update(layers::Layer &layer) override;
        private:
            double beta1;
            double beta2;
            double epsilon;
            std::unordered_map<std::string, Matrix> m;
            std::unordered_map<std::string, Matrix> v;
            int t;
    };
    class AdaGrad : public Optimizer {
        public:
            AdaGrad(double learningRate = 0.01, double epsilon = 1e-8);
            void update(layers::Layer &layer) override;
        private:
            double epsilon;
            std::unordered_map<std::string, Matrix> v;
    };
    class RMSProp : public Optimizer {
        public:
            RMSProp(double learningRate = 0.01, double beta = 0.9, double epsilon = 1e-8);
            void update(layers::Layer &layer) override;
        private:
            double beta;
            double epsilon;
            std::unordered_map<std::string, Matrix> v;
    };
}

#endif
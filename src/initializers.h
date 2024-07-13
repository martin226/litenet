#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include "matrix.h"

#include <random>
#include <string>

namespace litenet::initializers {
    class Initializer {
        public:
            Initializer();
            virtual ~Initializer() {}
            virtual Matrix initialize(int rows, int cols) const = 0;
    };
    class Zeros : public Initializer {
        public:
            Zeros();
            Matrix initialize(int rows, int cols) const override;
    };
    class Ones : public Initializer {
        public:
            Ones();
            Matrix initialize(int rows, int cols) const override;
    };
    class RandomInitializer : public Initializer {
        public:
            RandomInitializer();
            virtual ~RandomInitializer() {}
            virtual Matrix initialize(int rows, int cols) const = 0;
        protected:
            mutable std::random_device rd;
            mutable std::mt19937 gen;
    };
    class RandomUniform : public RandomInitializer {
        public:
            RandomUniform(double min = 0.0, double max = 1.0);
            Matrix initialize(int rows, int cols) const override;
        private:
            double min;
            double max;
    };
    class RandomNormal : public RandomInitializer {
        public:
            RandomNormal(double mean = 0.0, double stddev = 1);
            Matrix initialize(int rows, int cols) const override;
        private:
            double mean;
            double stddev;
    };
    class GlorotUniform : public RandomInitializer {
        public:
            GlorotUniform();
            Matrix initialize(int rows, int cols) const override;
    };
    class GlorotNormal : public RandomInitializer {
        public:
            GlorotNormal();
            Matrix initialize(int rows, int cols) const override;
    };
    class HeUniform : public RandomInitializer {
        public:
            HeUniform();
            Matrix initialize(int rows, int cols) const override;
    };    
    class HeNormal : public RandomInitializer {
        public:
            HeNormal();
            Matrix initialize(int rows, int cols) const override;
    };
}

#endif
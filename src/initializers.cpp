#include "initializers.h"
#include <random>

namespace litenet::initializers {
    Initializer::Initializer() {}

    Zeros::Zeros() {}

    Matrix Zeros::initialize(int rows, int cols) const {
        Matrix result(rows, cols, 0);
        return result;
    }

    Ones::Ones() {}

    Matrix Ones::initialize(int rows, int cols) const {
        Matrix result(rows, cols, 1);
        return result;
    }

    RandomInitializer::RandomInitializer() : gen(rd()) {}

    RandomUniform::RandomUniform(double min, double max) : min(min), max(max) {}

    Matrix RandomUniform::initialize(int rows, int cols) const {
        std::uniform_real_distribution<double> dist(min, max);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    RandomNormal::RandomNormal(double mean, double stddev) : mean(mean), stddev(stddev) {}

    Matrix RandomNormal::initialize(int rows, int cols) const {
        std::normal_distribution<double> dist(mean, stddev);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    GlorotUniform::GlorotUniform() {}

    Matrix GlorotUniform::initialize(int rows, int cols) const {
        double limit = std::sqrt(6.0 / (rows + cols));
        std::uniform_real_distribution<double> dist(-limit, limit);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    GlorotNormal::GlorotNormal() {}

    Matrix GlorotNormal::initialize(int rows, int cols) const {
        double stddev = std::sqrt(2.0 / (rows + cols));
        std::normal_distribution<double> dist(0, stddev);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    HeUniform::HeUniform() {}

    Matrix HeUniform::initialize(int rows, int cols) const {
        double limit = std::sqrt(6.0 / rows);
        std::uniform_real_distribution<double> dist(-limit, limit);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    HeNormal::HeNormal() {}

    Matrix HeNormal::initialize(int rows, int cols) const {
        double stddev = std::sqrt(2.0 / rows);
        std::normal_distribution<double> dist(0, stddev);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }
}
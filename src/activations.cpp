#include "activations.h"
#include <cmath>

namespace litenet::activations {
    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    Matrix sigmoid(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = sigmoid(m(i, j));
            }
        }
        return result;
    }

    Matrix sigmoidPrime(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                double s = sigmoid(m(i, j));
                result(i, j) = s * (1 - s);
            }
        }
        return result;
    }

    double relu(double x) {
        return x > 0 ? x : 0;
    }

    Matrix relu(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = relu(m(i, j));
            }
        }
        return result;
    }

    Matrix reluPrime(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = m(i, j) > 0 ? 1 : 0;
            }
        }
        return result;
    }

    double leakyRelu(double x, double negativeSlope) {
        return x > 0 ? x : negativeSlope * x;
    }

    Matrix leakyRelu(const Matrix &m, double negativeSlope) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = leakyRelu(m(i, j), negativeSlope);
            }
        }
        return result;
    }

    Matrix leakyReluPrime(const Matrix &m, double negativeSlope) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = m(i, j) > 0 ? 1 : negativeSlope;
            }
        }
        return result;
    }

    Matrix tanh(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = std::tanh(m(i, j));
            }
        }
        return result;
    }

    Matrix tanhPrime(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                double t = std::tanh(m(i, j));
                result(i, j) = 1 - t * t;
            }
        }
        return result;
    }

    Matrix softmax(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            double max = m(i, 0); // find max value in row
            for (int j = 1; j < m.getCols(); j++) {
                if (m(i, j) > max) {
                    max = m(i, j);
                }
            }
            double sum = 0; // compute sum of exponential
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = exp(m(i, j) - max);
                sum += result(i, j);
            }

            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) /= sum;
            }
        }
        return result;
    }

    Matrix softmaxPrime(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            double max = m(i, 0); // find max value in row
            for (int j = 1; j < m.getCols(); j++) {
                if (m(i, j) > max) {
                    max = m(i, j);
                }
            }
            double sum = 0; // compute sum of exponential
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = exp(m(i, j) - max);
                sum += result(i, j);
            }
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) /= sum;
                result(i, j) *= (1 - result(i, j)); // softmax prime is softmax * (1 - softmax)
            }
        }
        return result;
    }

    double linear(double x) {
        return x;
    }

    Matrix linear(const Matrix &m) {
        return m;
    }

    Matrix linearPrime(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        result.fill(1);
        return result;
    }
}
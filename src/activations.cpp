#include "activations.h"
#include <cmath>

namespace litenet::activations {
    Matrix sigmoid(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = 1 / (1 + exp(-m(i, j)));
            }
        }
        return result;
    }

    Matrix relu(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = m(i, j) > 0 ? m(i, j) : 0;
            }
        }
        return result;
    }

    Matrix leakyRelu(const Matrix &m, double negativeSlope) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = m(i, j) > 0 ? m(i, j) : negativeSlope * m(i, j);
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

    Matrix softmax(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            double max = m(i, 0);
            for (int j = 1; j < m.getCols(); j++) {
                if (m(i, j) > max) {
                    max = m(i, j);
                }
            }
            double sum = 0;
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

    Matrix linear(const Matrix &m) {
        return m;
    }

    Matrix sigmoidDerivative(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                double s = 1 / (1 + exp(-m(i, j)));
                result(i, j) = s * (1 - s);
            }
        }
        return result;
    }

    Matrix reluDerivative(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = m(i, j) > 0 ? 1 : 0;
            }
        }
        return result;
    }

    Matrix leakyReluDerivative(const Matrix &m, double negativeSlope) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                result(i, j) = m(i, j) > 0 ? 1 : negativeSlope;
            }
        }
        return result;
    }

    Matrix tanhDerivative(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                double t = std::tanh(m(i, j));
                result(i, j) = 1 - t * t;
            }
        }
        return result;
    }

    Matrix softmaxDerivative(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                double s = 1 / (1 + exp(-m(i, j)));
                result(i, j) = s * (1 - s);
            }
        }
        return result;
    }

    Matrix linearDerivative(const Matrix &m) {
        Matrix result(m.getRows(), m.getCols());
        result.fill(1);
        return result;
    }
}
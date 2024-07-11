#include "loss.h"
#include <cmath>
#include <stdexcept>

namespace litenet::loss {
    double meanSquaredError(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("predictions and targets must have the same shape");
        }
        return (predictions - targets).pow(2).sum() / predictions.getRows();
    }

    Matrix meanSquaredErrorPrime(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("predictions and targets must have the same shape");
        }
        return 2 * (predictions - targets) / predictions.getRows();
    }

    double meanAbsoluteError(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("predictions and targets must have the same shape");
        }
        return (predictions - targets).abs().sum() / predictions.getRows();
    }

    Matrix meanAbsoluteErrorPrime(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("predictions and targets must have the same shape");
        }
        return (predictions - targets).sign() / predictions.getRows();
    }

    double binaryCrossentropy(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("predictions and targets must have the same shape");
        }
        double crossentropy = 0;
        const double epsilon = 1e-7;
        for (int i = 0; i < predictions.getRows(); i++) {
            for (int j = 0; j < predictions.getCols(); j++) {
                double p = predictions(i, j);
                double t = targets(i, j);
                crossentropy += t * std::log(p + epsilon) + (1 - t) * std::log(1 - p + epsilon);
            }
        }
        return -crossentropy / predictions.getRows();
    }

    Matrix binaryCrossentropyPrime(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("predictions and targets must have the same shape");
        }
        Matrix result(predictions.getRows(), predictions.getCols());
        const double epsilon = 1e-7;
        for (int i = 0; i < predictions.getRows(); i++) {
            for (int j = 0; j < predictions.getCols(); j++) {
                double p = predictions(i, j);
                double t = targets(i, j);
                result(i, j) = -(t / (p + epsilon)) + ((1 - t) / (1 - p + epsilon));
            }
        }
        return result / predictions.getRows();
    }

    double categoricalCrossentropy(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("predictions and targets must have the same shape");
        }
        double crossentropy = 0;
        const double epsilon = 1e-7;
        for (int i = 0; i < predictions.getRows(); i++) {
            for (int j = 0; j < predictions.getCols(); j++) {
                double p = predictions(i, j);
                double t = targets(i, j);
                crossentropy += t * std::log(p + epsilon);
            }
        }
        return -crossentropy / predictions.getRows();
    }

    Matrix categoricalCrossentropyPrime(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("predictions and targets must have the same shape");
        }
        Matrix result(predictions.getRows(), predictions.getCols());
        const double epsilon = 1e-7;
        for (int i = 0; i < predictions.getRows(); i++) {
            for (int j = 0; j < predictions.getCols(); j++) {
                double p = predictions(i, j);
                double t = targets(i, j);
                result(i, j) = -t / (p + epsilon);
            }
        }
        return result / predictions.getRows();
    }
}
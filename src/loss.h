#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

namespace litenet::loss {
        double meanSquaredError(const Matrix &predictions, const Matrix &targets);
        Matrix meanSquaredErrorPrime(const Matrix &predictions, const Matrix &targets);
        double meanAbsoluteError(const Matrix &predictions, const Matrix &targets);
        Matrix meanAbsoluteErrorPrime(const Matrix &predictions, const Matrix &targets);
        double binaryCrossentropy(const Matrix &predictions, const Matrix &targets);
        Matrix binaryCrossentropyPrime(const Matrix &predictions, const Matrix &targets);
        double categoricalCrossentropy(const Matrix &predictions, const Matrix &targets);
        Matrix categoricalCrossentropyPrime(const Matrix &predictions, const Matrix &targets);
}

#endif
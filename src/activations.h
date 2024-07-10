#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "matrix.h"
#include <vector>

namespace litenet::activations {
    Matrix sigmoid(const Matrix &m);
    Matrix relu(const Matrix &m);
    Matrix leakyRelu(const Matrix &m, double negativeSlope = 0.2);
    Matrix tanh(const Matrix &m);
    Matrix softmax(const Matrix &m);
    Matrix linear(const Matrix &m);

    Matrix sigmoidPrime(const Matrix &m);
    Matrix reluPrime(const Matrix &m);
    Matrix leakyReluPrime(const Matrix &m, double negativeSlope = 0.2);
    Matrix tanhPrime(const Matrix &m);
    Matrix softmaxPrme(const Matrix &m);
    Matrix linearPrime(const Matrix &m);
}

#endif
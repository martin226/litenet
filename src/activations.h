#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "matrix.h"
#include <vector>

namespace litenet::activations {
    double sigmoid(double x);
    Matrix sigmoid(const Matrix &m);
    Matrix sigmoidPrime(const Matrix &m);

    double relu(double x);
    Matrix relu(const Matrix &m);
    Matrix reluPrime(const Matrix &m);

    double leakyRelu(double x, double negativeSlope = 0.2);
    Matrix leakyRelu(const Matrix &m, double negativeSlope = 0.2);
    Matrix leakyReluPrime(const Matrix &m, double negativeSlope = 0.2);

    Matrix tanh(const Matrix &m);
    Matrix tanhPrime(const Matrix &m);

    Matrix softmax(const Matrix &m);
    Matrix softmaxPrime(const Matrix &m);

    double linear(double x);    
    Matrix linear(const Matrix &m);
    Matrix linearPrime(const Matrix &m);
}

#endif
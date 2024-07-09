#include "matrix.h"

#include <iostream>
#include <random>

litenet::Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols) {}

litenet::Matrix::Matrix(int rows, int cols, bool random) : rows(rows), cols(cols), data(rows * cols) {
    if (random) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0, 1);
        for (size_t i = 0; i < rows * cols; i++) {
            data[i] = dist(gen);
        }
    }
}

litenet::Matrix::Matrix(const Matrix &m) : rows(m.rows), cols(m.cols), data(m.data) {}

litenet::Matrix::~Matrix() {
    data.clear();
}

litenet::Matrix &litenet::Matrix::operator=(const Matrix &m) {
    if (this != &m) {
        rows = m.rows;
        cols = m.cols;
        data = m.data;
    }
    return *this;
}

double &litenet::Matrix::operator()(int i, int j) {
    return data[i * cols + j];
}

double litenet::Matrix::operator()(int i, int j) const {
    return data[i * cols + j];
}

int litenet::Matrix::getRows() const {
    return rows;
}

int litenet::Matrix::getCols() const {
    return cols;
}

litenet::Matrix litenet::Matrix::operator+(const Matrix &m) const { // Element-wise addition
    if (rows != m.rows || cols != m.cols) {
        throw std::invalid_argument("Matrix dimensions are not compatible for addition");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] + m.data[i];
    }
    return result;
}

litenet::Matrix litenet::Matrix::operator-(const Matrix &m) const { // Element-wise subtraction
    if (rows != m.rows || cols != m.cols) {
        throw std::invalid_argument("Matrix dimensions are not compatible for subtraction");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] - m.data[i];
    }
    return result;
}

litenet::Matrix litenet::Matrix::operator*(const Matrix &m) const { // Matrix multiplication (dot product)
    if (cols != m.rows) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }
    Matrix result(rows, m.cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            for (size_t k = 0; k < cols; k++) {
                result(i, j) += data[i * cols + k] * m(k, j);
            }
        }
    }
    return result;
}

litenet::Matrix litenet::Matrix::operator*(double factor) const { // Scalar multiplication
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] * factor;
    }
    return result;
}

litenet::Matrix litenet::Matrix::operator/(double factor) const { // Scalar division
    if (factor == 0) {
        throw std::invalid_argument("Division by zero");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] / factor;
    }
    return result;
}

litenet::Matrix litenet::Matrix::operator-() const { // Unary minus
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        result.data[i] = -data[i];
    }
    return result;
}

litenet::Matrix &litenet::Matrix::operator+=(const Matrix &m) { // Element-wise addition assignment
    if (rows != m.rows || cols != m.cols) {
        throw std::invalid_argument("Matrix dimensions are not compatible for addition");
    }
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] += m.data[i];
    }
    return *this;
}

litenet::Matrix &litenet::Matrix::operator-=(const Matrix &m) { // Element-wise subtraction assignment
    if (rows != m.rows || cols != m.cols) {
        throw std::invalid_argument("Matrix dimensions are not compatible for subtraction");
    }
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] -= m.data[i];
    }
    return *this;
}

litenet::Matrix &litenet::Matrix::operator*=(const Matrix &m) { // Matrix multiplication assignment
    if (cols != m.rows) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }
    Matrix result(rows, m.cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            for (size_t k = 0; k < cols; k++) {
                result(i, j) += data[i * cols + k] * m(k, j);
            }
        }
    }
    *this = result;
    return *this;
}

litenet::Matrix &litenet::Matrix::operator*=(double factor) { // Scalar multiplication assignment
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] *= factor;
    }
    return *this;
}

litenet::Matrix &litenet::Matrix::operator/=(double factor) { // Scalar division assignment
    if (factor == 0) {
        throw std::invalid_argument("Division by zero");
    }
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] /= factor;
    }
    return *this;
}

bool litenet::Matrix::operator==(const Matrix &m) const {
    if (rows != m.rows || cols != m.cols) {
        return false;
    }
    for (size_t i = 0; i < rows * cols; i++) {
        if (data[i] != m.data[i]) {
            return false;
        }
    }
    return true;
}

bool litenet::Matrix::operator!=(const Matrix &m) const {
    return !(*this == m);
}

void litenet::Matrix::transpose() {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result(j, i) = data[i * cols + j];
        }
    }
    *this = result;
}

litenet::Matrix litenet::Matrix::transposed() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result(j, i) = data[i * cols + j];
        }
    }
    return result;
}

void litenet::Matrix::normalize() {
    double s = sum();
    if (s == 0) {
        throw std::invalid_argument("Normalization of zero vector");
    }
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] /= s;
    }
}

litenet::Matrix litenet::Matrix::normalized() const {
    double s = sum();
    if (s == 0) {
        throw std::invalid_argument("Normalization of zero vector");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] / s;
    }
    return result;
}

double litenet::Matrix::sum() const {
    double s = 0;
    for (size_t i = 0; i < rows * cols; i++) {
        s += data[i];
    }
    return s;
}

std::vector<double> litenet::Matrix::flatten() const {
    return data;
}

litenet::Matrix litenet::Matrix::reshape(const std::vector<double> &v, int rows, int cols) {
    if (v.size() != rows * cols) {
        throw std::invalid_argument("Invalid vector size for reshaping");
    }
    Matrix result(rows, cols);
    result.data = v;
    return result;
}

void litenet::Matrix::print() const {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
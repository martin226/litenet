#include "matrix.h"

#include <iostream>
#include <random>

namespace litenet {
    Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, 0) {}

    Matrix::Matrix(int rows, int cols, bool random) : rows(rows), cols(cols), data(rows * cols) {
        if (random) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0, 1);
            for (size_t i = 0; i < rows * cols; i++) {
                data[i] = dist(gen);
            }
        }
    }

    Matrix::Matrix(const Matrix &m) : rows(m.rows), cols(m.cols), data(m.data) {}

    Matrix::~Matrix() {
        data.clear();
    }

    Matrix &Matrix::operator=(const Matrix &m) {
        if (this != &m) {
            rows = m.rows;
            cols = m.cols;
            data = m.data;
        }
        return *this;
    }

    double &Matrix::operator()(int i, int j) {
        return data[i * cols + j];
    }

    double Matrix::operator()(int i, int j) const {
        return data[i * cols + j];
    }

    int Matrix::getRows() const {
        return rows;
    }

    int Matrix::getCols() const {
        return cols;
    }

    Matrix Matrix::operator+(const Matrix &m) const { // Element-wise addition
        if (rows != m.rows || cols != m.cols) {
            throw std::invalid_argument("Matrix dimensions are not compatible for addition");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] + m.data[i];
        }
        return result;
    }

    Matrix Matrix::operator+(double scalar) const { // Scalar addition
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] + scalar;
        }
        return result;
    }

    Matrix operator+(double scalar, const Matrix &m) { // Scalar addition
        return m + scalar;
    }

    Matrix Matrix::operator-(const Matrix &m) const { // Element-wise subtraction
        if (rows != m.rows || cols != m.cols) {
            throw std::invalid_argument("Matrix dimensions are not compatible for subtraction");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] - m.data[i];
        }
        return result;
    }

    Matrix Matrix::operator-(double scalar) const { // Scalar subtraction
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] - scalar;
        }
        return result;
    }

    Matrix operator-(double scalar, const Matrix &m) { // Scalar subtraction
        return -m + scalar;
    }

    Matrix Matrix::operator*(const Matrix &m) const { // Matrix multiplication (dot product)
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

    Matrix Matrix::operator*(double factor) const { // Scalar multiplication
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] * factor;
        }
        return result;
    }

    Matrix operator*(double factor, const Matrix &m) { // Scalar multiplication
        return m * factor;
    }    

    Matrix Matrix::operator/(double factor) const { // Scalar division
        if (factor == 0) {
            throw std::invalid_argument("Division by zero");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] / factor;
        }
        return result;
    }

    Matrix Matrix::operator-() const { // Unary minus
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = -data[i];
        }
        return result;
    }

    Matrix &Matrix::operator+=(const Matrix &m) { // Element-wise addition assignment
        if (rows != m.rows || cols != m.cols) {
            throw std::invalid_argument("Matrix dimensions are not compatible for addition");
        }
        for (size_t i = 0; i < rows * cols; i++) {
            data[i] += m.data[i];
        }
        return *this;
    }

    Matrix &Matrix::operator+=(double scalar) { // Scalar addition assignment
        for (size_t i = 0; i < rows * cols; i++) {
            data[i] += scalar;
        }
        return *this;
    }

    Matrix &Matrix::operator-=(const Matrix &m) { // Element-wise subtraction assignment
        if (rows != m.rows || cols != m.cols) {
            throw std::invalid_argument("Matrix dimensions are not compatible for subtraction");
        }
        for (size_t i = 0; i < rows * cols; i++) {
            data[i] -= m.data[i];
        }
        return *this;
    }

    Matrix &Matrix::operator-=(double scalar) { // Scalar subtraction assignment
        for (size_t i = 0; i < rows * cols; i++) {
            data[i] -= scalar;
        }
        return *this;
    }

    Matrix &Matrix::operator*=(const Matrix &m) { // Matrix multiplication assignment
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

    Matrix &Matrix::operator*=(double factor) { // Scalar multiplication assignment
        for (size_t i = 0; i < rows * cols; i++) {
            data[i] *= factor;
        }
        return *this;
    }

    Matrix &Matrix::operator/=(double factor) { // Scalar division assignment
        if (factor == 0) {
            throw std::invalid_argument("Division by zero");
        }
        for (size_t i = 0; i < rows * cols; i++) {
            data[i] /= factor;
        }
        return *this;
    }

    bool Matrix::operator==(const Matrix &m) const {
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

    bool Matrix::operator!=(const Matrix &m) const {
        return !(*this == m);
    }

    Matrix Matrix::hadamard(const Matrix &m) const {
        if (rows != m.rows || cols != m.cols) {
            throw std::invalid_argument("Matrix dimensions are not compatible for Hadamard product");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] * m.data[i];
        }
        return result;
    }

    Matrix Matrix::transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(j, i) = data[i * cols + j];
            }
        }
        return result;
    }

    Matrix Matrix::normalize() const {
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

    Matrix Matrix::pow(double exponent) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = std::pow(data[i], exponent);
        }
        return result;
    }

    Matrix Matrix::abs() const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = std::abs(data[i]);
        }
        return result;
    }

    Matrix Matrix::sign() const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] > 0 ? 1 : data[i] < 0 ? -1 : 0;
        }
        return result;
    }

    Matrix Matrix::log(double base = 2) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = std::log(data[i]) / std::log(base);
        }
        return result;
    }

    double Matrix::Matrix::sum() const {
        double s = 0;
        for (size_t i = 0; i < rows * cols; i++) {
            s += data[i];
        }
        return s;
    }

    Matrix Matrix::Matrix::sum(int axis) const {
        if (axis == 0) { // Sum along columns
            Matrix result(1, cols);
            for (size_t j = 0; j < cols; j++) {
                for (size_t i = 0; i < rows; i++) {
                    result(0, j) += data[i * cols + j];
                }
            }
            return result;
        } else if (axis == 1) { // Sum along rows
            Matrix result(rows, 1);
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    result(i, 0) += data[i * cols + j];
                }
            }
            return result;
        } else {
            throw std::invalid_argument("Invalid axis for sum");
        }
    }

    std::vector<double> Matrix::Matrix::flatten() const {
        return data;
    }

    Matrix Matrix::reshape(const std::vector<double> &v, int rows, int cols) {
        if (v.size() != rows * cols) {
            throw std::invalid_argument("Invalid vector size for reshaping");
        }
        Matrix result(rows, cols);
        result.data = v;
        return result;
    }

    void Matrix::fill(double value) {
        for (size_t i = 0; i < rows * cols; i++) {
            data[i] = value;
        }
    }

    Matrix Matrix::subsetCols(int start, int end) const {
        if (start < 0 || start >= cols || end < 0 || end >= cols || start > end) {
            throw std::invalid_argument("Invalid column subset");
        }
        Matrix result(rows, end - start + 1);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = start; j <= end; j++) {
                result(i, j - start) = data[i * cols + j];
            }
        }
        return result;
    }

    Matrix Matrix::subsetRows(int start, int end) const {
        if (start < 0 || start >= rows || end < 0 || end >= rows || start > end) {
            throw std::invalid_argument("Invalid row subset");
        }
        Matrix result(end - start + 1, cols);
        for (size_t i = start; i <= end; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(i - start, j) = data[i * cols + j];
            }
        }
        return result;
    }

    void Matrix::print() const {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}
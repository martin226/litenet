#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

namespace litenet {
    class Matrix {
        public:
            Matrix();
            Matrix(int rows, int cols);
            Matrix(int rows, int cols, double value);
            Matrix(const Matrix &m);
            ~Matrix();
            Matrix &operator=(const Matrix &m);
            double &operator()(int i, int j);
            double operator()(int i, int j) const;
            int getRows() const;
            int getCols() const;
            std::vector<int> getShape() const;
            Matrix operator+(const Matrix &m) const;
            Matrix operator+(double scalar) const;
            friend Matrix operator+(double scalar, const Matrix &m);
            Matrix operator-(const Matrix &m) const;
            Matrix operator-(double scalar) const;
            friend Matrix operator-(double scalar, const Matrix &m);
            Matrix operator*(const Matrix &m) const;
            Matrix operator*(double factor) const;
            friend Matrix operator*(double factor, const Matrix &m);
            Matrix operator/(double factor) const;
            Matrix operator/(const Matrix &m) const;
            Matrix operator-() const;
            Matrix &operator+=(const Matrix &m);
            Matrix &operator+=(double scalar);
            Matrix &operator-=(const Matrix &m);
            Matrix &operator-=(double scalar);
            Matrix &operator*=(const Matrix &m);
            Matrix &operator*=(double factor);
            Matrix &operator/=(double factor);
            Matrix &operator/=(const Matrix &m);
            bool operator==(const Matrix &m) const;
            bool operator!=(const Matrix &m) const;
            Matrix hadamard(const Matrix &m) const;
            Matrix transpose() const;
            Matrix normalize() const;
            Matrix pow(double exponent) const;
            Matrix abs() const;
            Matrix sign() const;
            Matrix log(double base = 2) const;
            double sum() const;
            Matrix sum(int axis) const;
            double max() const;
            Matrix max(int axis) const;
            double min() const;
            Matrix min(int axis) const;
            std::vector<double> flatten() const;
            static Matrix reshape(const std::vector<double> &v, int rows, int cols);
            void fill(double value);
            Matrix subsetCols(int start, int end) const;
            Matrix subsetRows(int start, int end) const;
            void swapRows(int i, int j);
            void swapCols(int i, int j);
            void print() const;
        private:
            int rows;
            int cols;
            std::vector<double> data;
    };
}

#endif
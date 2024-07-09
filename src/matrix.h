#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

namespace litenet {
    class Matrix {
        public:
            Matrix(int rows, int cols);
            Matrix(int rows, int cols, bool random);
            Matrix(const Matrix &m);
            ~Matrix();
            Matrix &operator=(const Matrix &m);
            double &operator()(int i, int j);
            double operator()(int i, int j) const;
            int getRows() const;
            int getCols() const;
            Matrix operator+(const Matrix &m) const;
            Matrix operator-(const Matrix &m) const;
            Matrix operator*(const Matrix &m) const;
            Matrix operator*(double factor) const;
            Matrix operator/(double factor) const;
            Matrix operator-() const;
            Matrix &operator+=(const Matrix &m);
            Matrix &operator-=(const Matrix &m);
            Matrix &operator*=(const Matrix &m);
            Matrix &operator*=(double factor);
            Matrix &operator/=(double factor);
            bool operator==(const Matrix &m) const;
            bool operator!=(const Matrix &m) const;
            void transpose();
            Matrix transposed() const;
            void normalize();
            Matrix normalized() const;
            double sum() const;
            std::vector<double> flatten() const;
            static Matrix reshape(const std::vector<double> &v, int rows, int cols);
            void print() const;
        private:
            int rows;
            int cols;
            std::vector<double> data;
    };
}

#endif
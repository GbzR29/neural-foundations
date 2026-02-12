/**
 * @file Matrix.hpp
 * @brief Basic linear algebra operations for neural network computations.
 * @author Gabriel Carvalho
 * @date 2026
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <fstream>

/**
 * @class Matrix
 * @brief Representation of a mathematical matrix and its operations.
 * Contiguous memory allocation via flattened std::vector for cache performance optimization.
 */
class Matrix
{
public:
    int rows, cols;
    std::vector<double> data;

    /**
     * @brief Construction of a new Matrix object.
     * @param r Number of rows.
     * @param c Number of columns.
     */
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    /**
     * @brief Element access at row 'r' and column 'c'.
     * @param r Row index.
     * @param c Column index.
     * @return double& Reference to the element.
     */
    double &operator()(int r, int c)
    {
        return data[r * cols + c];
    }

    /**
     * @brief Constant element access at row 'r' and column 'c'.
     */
    const double &operator()(int r, int c) const
    {
        return data[r * cols + c];
    }

    /**
     * @brief Matrix multiplication (Result = A * B).
     * @throws std::invalid_argument If dimensions are incompatible for multiplication.
     * @return Matrix Resulting matrix from the operation.
     */
    static Matrix multiply(const Matrix &a, const Matrix &b)
    {
        if (a.cols != b.rows)
        {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication.");
        }

        Matrix result(a.rows, b.cols);
        for (int i = 0; i < a.rows; ++i)
        {
            for (int k = 0; k < a.cols; ++k)
            {
                double temp = a(i, k);
                for (int j = 0; j < b.cols; ++j)
                {
                    result(i, j) += temp * b(k, j);
                }
            }
        }
        return result;
    }

    /**
     * @brief Element-wise addition of another matrix to the current instance.
     * @throws std::invalid_argument If dimensions are not identical.
     */
    void add(const Matrix &other)
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::invalid_argument("Matrix dimensions must match for addition.");
        }
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] += other.data[i];
        }
    }

    /**
     * @brief Transposition of the matrix.
     * @return Matrix Transposed matrix (rows and columns swapped).
     */
    Matrix transpose() const
    {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Application of a function to every element in the matrix.
     * Useful for activation functions across a whole layer.
     * @param func Transformation function (double to double).
     */
    void map(std::function<double(double)> func)
    {
        for (double &val : data)
        {
            val = func(val);
        }
    }

    /**
     * @brief Output of matrix contents to console for debugging purposes.
     */
    void print() const
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

    /**
     * @brief Serialization of matrix data to a binary file.
     * @param out Reference to the output file stream.
     */
    void save(std::ofstream &out) const
    {
        out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(double));
    }

    /**
     * @brief Deserialization of matrix data from a binary file.
     * @param in Reference to the input file stream.
     */
    void load(std::ifstream &in)
    {
        in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
        data.resize(rows * cols);
        in.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(double));
    }
};

#endif // MATRIX_HPP
/**
 * @file Activation.hpp
 * @brief Activation functions and their derivatives for neural network layers.
 * @author Gabriel Carvalho
 * @date 2026
 */

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <vector>
#include <cmath>
#include <algorithm>

/**
 * @namespace Activation
 * @brief Contains mathematical non-linear functions to be applied to neuron outputs.
 */
namespace Activation {

    /**
     * @brief Sigmoid activation function.
     * Maps any real value to a range between (0, 1).
     * Formula: 1 / (1 + exp(-z))
     * @param z Input value (weighted sum).
     * @return double Activated value.
     */
    inline double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    /**
     * @brief Derivative of the Sigmoid function.
     * Uses the pre-calculated sigmoid output for efficiency.
     * Formula: a * (1 - a)
     * @param a The output of the sigmoid function.
     * @return double Gradient value.
     */
    inline double sigmoid_derivative(double a) {
        return a * (1.0 - a);
    }

    /**
     * @brief ReLU (Rectified Linear Unit) activation function.
     * Returns 0 for negative inputs and the value itself for positive inputs.
     * Formula: max(0, z)
     * @param z Input value.
     * @return double Activated value.
     */
    inline double relu(double z) {
        return std::max(0.0, z);
    }

    /**
     * @brief Derivative of the ReLU function.
     * @param z Input value.
     * @return double 1.0 if z > 0, otherwise 0.0.
     */
    inline double relu_derivative(double z) {
        return (z > 0) ? 1.0 : 0.0;
    }

    /**
     * @brief Hyperbolic Tangent (Tanh) activation function.
     * Maps input to a range between (-1, 1).
     * @param z Input value.
     * @return double Activated value.
     */
    inline double tanh_activation(double z) {
        return std::tanh(z);
    }

    /**
     * @brief Derivative of the Tanh function.
     * Formula: 1 - a^2
     * @param a The output of the tanh function.
     * @return double Gradient value.
     */
    inline double tanh_derivative(double a) {
        return 1.0 - (a * a);
    }

    /**
     * @brief Leaky ReLU activation function.
     * Prevents "Dying ReLU" by allowing a small slope for negative values.
     * @param z Input value.
     * @param alpha Small constant for the negative slope (default: 0.01).
     * @return double Activated value.
     */
    inline double leaky_relu(double z, double alpha = 0.01) {
        return (z > 0) ? z : alpha * z;
    }

    /**
     * @brief Derivative of the Leaky ReLU function.
     * @param z Input value.
     * @param alpha Small constant for the negative slope.
     * @return double 1.0 if z > 0, otherwise alpha.
     */
    inline double leaky_relu_derivative(double z, double alpha = 0.01) {
        return (z > 0) ? 1.0 : alpha;
    }

    /**
     * @brief Softmax activation function.
     * Normalizes a vector into a probability distribution where the sum equals 1.0.
     * Includes numerical stability by subtracting the maximum value.
     * @param z Reference to a vector of raw scores (logits).
     * @return std::vector<double> Probability distribution.
     */
    inline std::vector<double> softmax(const std::vector<double>& z) {
        std::vector<double> a(z.size());
        double max_z = *std::max_element(z.begin(), z.end());
        double sum = 0.0;

        for (size_t i = 0; i < z.size(); ++i) {
            a[i] = std::exp(z[i] - max_z);
            sum += a[i];
        }

        for (size_t i = 0; i < z.size(); ++i) {
            a[i] /= sum;
        }
        return a;
    }
}

#endif // ACTIVATION_HPP
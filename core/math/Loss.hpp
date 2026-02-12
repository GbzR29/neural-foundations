/**
 * @file Loss.hpp
 * @brief Loss functions to measure the error between predicted and real values.
 * @author Gabriel Carvalho
 * @date 2026
 */

#ifndef LOSS_HPP
#define LOSS_HPP

#include <vector>
#include <cmath>
#include <stdexcept>

/**
 * @namespace Loss
 * @brief Functions to calculate the cost/error of the network.
 */
namespace Loss {

    /**
     * @brief Mean Squared Error (MSE).
     * Calculates the average of the squares of the errors.
     * @param target The ground truth (real values).
     * @param predicted The output from the network.
     * @return double The calculated error.
     */
    inline double mse(const std::vector<double>& target, const std::vector<double>& predicted) {
        if (target.size() != predicted.size()) {
            throw std::invalid_argument("Vector sizes must match for MSE calculation.");
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            double diff = target[i] - predicted[i];
            error += diff * diff;
        }
        return error / target.size();
    }

    /**
     * @brief Derivative of MSE with respect to the prediction.
     * Used during Backpropagation to know how to adjust weights.
     * Formula: 2 * (predicted - target) / n
     * @param target Real value.
     * @param predicted Predicted value.
     * @return double Gradient of the error.
     */
    inline double mse_derivative(double target, double predicted) {
        return 2.0 * (predicted - target);
    }
}

#endif // LOSS_HPP
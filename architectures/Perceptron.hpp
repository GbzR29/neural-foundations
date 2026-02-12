/**
 * @file Perceptron.hpp
 * @brief Implementation of a single-layer perceptron.
 * @author Gabriel Carvalho
 * @date 2026
 */

#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "../../core/tensor/Matrix.hpp"
#include "../../core/math/Activation.hpp"
#include <vector>
#include <random>

/**
 * @class Perceptron
 * @brief A single neuron model for binary classification.
 */
class Perceptron {
private:
    std::vector<double> weights;
    double bias;
    double learning_rate;

public:
    /**
     * @brief Construct a new Perceptron object with random weights.
     * @param input_size Number of input features.
     * @param lr Learning rate for weight updates.
     */
    Perceptron(int input_size, double lr = 0.01) : learning_rate(lr) {
        // Initialize weights with small random values
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

        for (int i = 0; i < input_size; ++i) {
            weights.push_back(distribution(generator));
        }
        bias = distribution(generator);
    }

    /**
     * @brief Forward pass: computes the weighted sum and applies activation.
     * Formula: y = step(sum(x_i * w_i) + b)
     */
    int predict(const std::vector<double>& inputs) {
        double sum = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weights[i];
        }
        
        // Using a basic Step Function for the classic Perceptron
        return (sum > 0) ? 1 : 0;
    }

    /**
     * @brief Trains the perceptron using the Perceptron Learning Rule.
     * @param inputs Input vector.
     * @param target Desired output (0 or 1).
     */
    void train(const std::vector<double>& inputs, int target) {
        int prediction = predict(inputs);
        int error = target - prediction;

        if (error != 0) {
            // Update weights: w = w + (lr * error * x)
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] += learning_rate * error * inputs[i];
            }
            // Update bias: b = b + (lr * error)
            bias += learning_rate * error;
        }
    }
};

#endif // PERCEPTRON_HPP
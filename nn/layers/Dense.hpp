/**
 * @file Dense.hpp
 * @brief Implementation of a Fully Connected (Dense) layer with dynamic activation.
 * @author Gabriel Carvalho
 * @date 2026
 */

#ifndef DENSE_HPP
#define DENSE_HPP

#include "../../core/tensor/Matrix.hpp"
#include "../../core/math/Activation.hpp"
#include <random>
#include <ctime>

/**
 * @class Dense
 * @brief Represents a fully connected layer with configurable activation functions.
 */
class Dense {
public:
    enum ActivationType { SIGMOID, RELU, TANH };

    Matrix weights;
    Matrix bias;
    Matrix last_input;  
    Matrix last_output; 
    ActivationType activation;

    /**
     * @brief Construction of a Dense layer with random initialization.
     * @param input_size Number of neurons in the previous layer.
     * @param output_size Number of neurons in this layer.
     * @param act Type of activation function (default: SIGMOID).
     */
    Dense(int input_size, int output_size, ActivationType act = SIGMOID)
        : weights(output_size, input_size),
          bias(output_size, 1),
          last_input(0, 0),
          last_output(0, 0),
          activation(act)
    {
        static std::default_random_engine gen(static_cast<unsigned int>(time(0)));
        
        // Adjusting initialization based on activation (He vs Xavier)
        double variance = (activation == RELU) ? 2.0 / input_size : 1.0 / input_size;
        double limit = sqrt(3.0 * variance);
        
        std::uniform_real_distribution<double> dist(-limit, limit);

        for (auto &w : weights.data) w = dist(gen);
        for (auto &b : bias.data) b = 0.01; // Small constant bias for ReLU stability
    }

    /**
     * @brief Forward pass applying the chosen activation function.
     */
    Matrix forward(const Matrix &input) {
        last_input = input; 
        Matrix z = Matrix::multiply(weights, input);
        z.add(bias);

        // Apply mapping based on activation type
        switch (activation) {
            case RELU:    z.map(Activation::relu); break;
            case TANH:    z.map(Activation::tanh_activation); break;
            case SIGMOID: 
            default:      z.map(Activation::sigmoid); break;
        }

        last_output = z; 
        return z;
    }

    /**
     * @brief Backward pass using the specific derivative of the layer's activation.
     */
    Matrix backward(const Matrix &output_gradient, double learning_rate) {
        Matrix activation_grad = last_output;

        // Apply derivative mapping
        switch (activation) {
            case RELU:    activation_grad.map(Activation::relu_derivative); break;
            case TANH:    activation_grad.map(Activation::tanh_derivative); break;
            case SIGMOID: 
            default:      activation_grad.map(Activation::sigmoid_derivative); break;
        }

        Matrix delta(output_gradient.rows, 1);
        for (int i = 0; i < output_gradient.rows; ++i) {
            delta(i, 0) = output_gradient(i, 0) * activation_grad(i, 0);
        }

        Matrix input_T = last_input.transpose();
        Matrix weights_gradient = Matrix::multiply(delta, input_T);

        for (size_t i = 0; i < weights.data.size(); ++i) {
            weights.data[i] -= learning_rate * weights_gradient.data[i];
        }
        for (size_t i = 0; i < bias.data.size(); ++i) {
            bias.data[i] -= learning_rate * delta.data[i];
        }

        Matrix weights_T = weights.transpose();
        return Matrix::multiply(weights_T, delta);
    }
};

#endif // DENSE_HPP
/**
 * @file Sequential.hpp
 * @brief Container to stack multiple layers and manage forward/backward flow.
 * @author Gabriel Carvalho
 * @date 2026
 */

#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "../layers/Dense.hpp"
#include "../../core/math/Loss.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

/**
 * @class Sequential
 * @brief Model architecture for linear stacking of layers.
 */
class Sequential {
private:
    std::vector<Dense> layers;

public:
    /**
     * @brief Addition of a Dense layer to the model.
     * @param layer Dense layer instance.
     */
    void add(const Dense& layer) {
        layers.push_back(layer);
    }

    /**
     * @brief Prediction of output for a given input via forward pass.
     * @param input Matrix (input_size x 1).
     * @return Matrix Result of the computation.
     */
    Matrix predict(Matrix input) {
        Matrix current_output = input;
        for (auto& layer : layers) {
            current_output = layer.forward(current_output);
        }
        return current_output;
    }

    /**
     * @brief Training of the model on a single input-target pair.
     * @param input Input features matrix.
     * @param target Expected output matrix.
     * @param learning_rate Scaling factor for weight updates.
     */
    void train(const Matrix& input, const Matrix& target, double learning_rate) {
        // Forward Pass
        Matrix output = predict(input);

        // Initial Gradient Calculation (Output Error)
        Matrix gradient(target.rows, 1);
        for (int i = 0; i < target.rows; ++i) {
            gradient(i, 0) = Loss::mse_derivative(target(i, 0), output(i, 0));
        }

        // Backward Pass in reverse order
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            gradient = layers[i].backward(gradient, learning_rate);
        }
    }

    /**
     * @brief Serialization of the entire model to a .frac file.
     * @param filename Path for the output file.
     */
    void save_model(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) throw std::runtime_error("Could not open file for saving.");

        // Header identification: Magic Number
        const char magic[4] = {'F', 'R', 'A', 'C'};
        out.write(magic, 4);

        // Quantity of layers
        int num_layers = static_cast<int>(layers.size());
        out.write(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        // Binary storage of each layer
        for (const auto& layer : layers) {
            layer.weights.save(out);
            layer.bias.save(out);
        }
        out.close();
    }

    /**
     * @brief Deserialization of the model from a .frac file.
     * @param filename Path to the source file.
     */
    void load_model(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("Could not open model file.");

        // Magic Number verification
        char magic[4];
        in.read(magic, 4);
        if (magic[0] != 'F' || magic[1] != 'R' || magic[2] != 'A' || magic[3] != 'C') {
            throw std::runtime_error("Invalid .frac file format.");
        }

        int num_layers;
        in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        // Clear existing layers and reconstruct from file
        layers.clear();
        for (int i = 0; i < num_layers; ++i) {
            Matrix temp_w(0,0);
            Matrix temp_b(0,0);
            temp_w.load(in);
            temp_b.load(in);
            
            // Dense layer reconstruction using stored dimensions
            Dense layer(temp_w.cols, temp_w.rows);
            layer.weights = temp_w;
            layer.bias = temp_b;
            layers.push_back(layer);
        }
        in.close();
    }
};

#endif // SEQUENTIAL_HPP
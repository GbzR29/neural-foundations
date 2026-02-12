/**
 * @file TextEncoder.hpp
 * @brief Utility for converting strings into Matrix format for neural networks.
 * @author Gabriel Carvalho
 * @date 2026
 */

#ifndef TEXT_ENCODER_HPP
#define TEXT_ENCODER_HPP

#include "../../core/tensor/Matrix.hpp"
#include <string>
#include <vector>
#include <algorithm>

/**
 * @class TextEncoder
 * @brief Handlers string-to-matrix conversion using fixed-length ASCII normalization.
 */
class TextEncoder {
public:
    /**
     * @brief Conversion of a string to a normalized Matrix.
     * @param text The input string (e.g., "Gabriel").
     * @param max_length Fixed size of the output vector.
     * @return Matrix A (max_length x 1) matrix.
     */
    static Matrix encode(const std::string& text, int max_length) {
        Matrix encoded(max_length, 1);
        
        for (int i = 0; i < max_length; ++i) {
            if (i < static_cast<int>(text.length())) {
                // Normalization of ASCII value (0-255) to range [0, 1]
                encoded(i, 0) = static_cast<double>(static_cast<unsigned char>(text[i])) / 255.0;
            } else {
                // Padding with 0 for empty spaces
                encoded(i, 0) = 0.0;
            }
        }
        
        return encoded;
    }

    /**
     * @brief Decoding of a target index to a category name.
     * @param index The neuron index with highest activation.
     * @param categories Vector of category names.
     * @return std::string The name of the predicted category.
     */
    static std::string decode_label(int index, const std::vector<std::string>& categories) {
        if (index >= 0 && index < static_cast<int>(categories.size())) {
            return categories[index];
        }
        return "Unknown";
    }
};

#endif // TEXT_ENCODER_HPP
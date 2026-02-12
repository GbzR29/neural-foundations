#include "nn/model/Sequential.hpp"
#include "nn/utils/TextEncoder.hpp"
#include <iostream>
#include <vector>
#include <string>

void run_inference(Sequential& model, 
                   const std::string& checkpoint_path, 
                   const std::string& word_to_test);

void train_model_logic(Sequential& model, 
                       const std::vector<std::string>& words, 
                       const std::vector<double>& labels, 
                       const std::string& checkpoint_path);

                       

int main() {
    Sequential model;
    model.add(Dense(10, 32, Dense::RELU)); 
    model.add(Dense(32, 1, Dense::SIGMOID));  

    // Dados misturados (Alternando Pessoa e Objeto)
    std::vector<std::string> words = {"Gabriel", "Lapis", "Maria", "Cadeira", "Renata", "Controle", "Michael", "Abajur"};
    std::vector<double> labels = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};

    double learning_rate = 0.005; // Bem menor para estabilidade
    int epochs = 10000;
    int max_len = 10;

    std::cout << "Treinando com aprendizado suave..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < words.size(); ++i) {
            Matrix input = TextEncoder::encode(words[i], max_len);
            Matrix target(1, 1);
            target(0, 0) = labels[i];
            model.train(input, target, learning_rate);
        }
        if (epoch % 2000 == 0) std::cout << "Progresso: " << (epoch*100/epochs) << "%" << std::endl;
    }

    model.save_model("checkpoints/V1.frac");
    
    // Teste
    std::cout << "\n--- RESULTADOS ---" << std::endl;
    run_inference(model, "checkpoints/V1.frac", "Gabriel");
    run_inference(model, "checkpoints/V1.frac", "Lapis");
}


/**
 * @brief Treina o modelo com um conjunto de dados e salva em um arquivo .frac.
 * @param model Referência para o modelo Sequential.
 * @param words Lista de palavras para treinamento.
 * @param labels Lista de categorias (0 para Pessoa, 1 para Objeto).
 * @param checkpoint_path Caminho para salvar o arquivo .frac.
 */
void train_model_logic(Sequential& model, 
                       const std::vector<std::string>& words, 
                       const std::vector<double>& labels, 
                       const std::string& checkpoint_path) {
    
    int epochs = 2000;
    double learning_rate = 0.01;
    int max_len = 10;

    std::cout << "Iniciando treinamento..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0;
        for (size_t i = 0; i < words.size(); ++i) {
            Matrix input = TextEncoder::encode(words[i], max_len);
            Matrix target(1, 1);
            target(0, 0) = labels[i];

            model.train(input, target, learning_rate);
        }
    }

    model.save_model(checkpoint_path);
    std::cout << "Modelo treinado e salvo em: " << checkpoint_path << std::endl;
}

/**
 * @brief Carrega um modelo existente e realiza uma predicao.
 * @param model Referencia para o modelo Sequential.
 * @param checkpoint_path Caminho do arquivo .frac.
 * @param word_to_test Palavra que a IA deve classificar.
 */
void run_inference(Sequential& model, 
                   const std::string& checkpoint_path, 
                   const std::string& word_to_test) {
    
    try {
        model.load_model(checkpoint_path);
        
        Matrix input = TextEncoder::encode(word_to_test, 10);
        Matrix result = model.predict(input);

        std::cout << "Analise da palavra '" << word_to_test << "':" << std::endl;
        std::cout << "Valor bruto: " << result(0, 0) << std::endl;

        if (result(0, 0) < 0.5) {
            std::cout << "Resultado: PESSOA" << std::endl;
        } else {
            std::cout << "Resultado: OBJETO" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Erro ao carregar modelo: " << e.what() << std::endl;
    }
}
#include <iostream>
#include <memory>
#include <vector>
#include "core/nodes.h"
#include "core/neural_network.h"

int main() {
    const int input_nodes = 3;
    const int hidden_nodes = 3;
    const int output_nodes = 1;
    const double learning_rate = 0.1;
    const int epochs = 100000;

    nodes node = { hidden_nodes, input_nodes, output_nodes };
    auto nn = std::make_unique<neural_network>(node, learning_rate);

    std::vector<std::vector<double>> inputs = {
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };

    const std::vector<std::vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };

    for (int i = 0; i < epochs; ++i) {
        for (size_t j = 0; j < inputs.size(); ++j) {
            nn->train(inputs[j], targets[j]);
        }

        if ((i + 1) % 1000 == 0) {
            printf("epoch %d complete\n", i + 1);
        }
    }

    const std::vector<std::vector<double>> testInputs = {
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };

    for (const auto& testInput : testInputs) {
        const auto result = nn->predict(testInput);

        printf("input: ");
        for (const double val : testInput) {
            printf("%f ", val);
        }

        printf("=> output: ");
        for (const double& val : result) {
            printf("%f ", val);
        }

        printf("\n");
    }

    return 0;
}

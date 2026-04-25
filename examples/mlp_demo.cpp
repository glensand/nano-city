#include "nn.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    // Learn a simple regression target: y = 0.5*x1 - x2 + 0.2
    const std::array<std::array<float, 2>, 8> xs{ {
        { -1.f, -1.f },
        { -1.f, 0.f },
        { 0.f, -1.f },
        { 0.f, 0.f },
        { 1.f, 0.f },
        { 0.f, 1.f },
        { 1.f, 1.f },
        { 2.f, -1.f },
    } };
    const std::array<float, 8> ys{ 0.7f, -0.3f, 1.2f, 0.2f, 0.7f, -0.8f, -0.3f, 2.2f };

    mlp net(2, { 8, 4, 1 });
    constexpr float lr = 0.05f;
    constexpr int steps = 800;
    bool loss_graph_dumped = false;

    for (int step = 0; step < steps; ++step) {
        float total_loss = 0.f;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::vector<scalar> input;
            input.reserve(2);
            input.emplace_back(xs[i][0], std::vector<scalar*>{}, "x1");
            input.emplace_back(xs[i][1], std::vector<scalar*>{}, "x2");

            const auto& pred = net.forward_one(input);
            auto loss = (pred + (-ys[i])).pow(2.f);

            net.zero_grad();
            loss.set_grad(1.f);
            loss.backward();

            if (!loss_graph_dumped) {
                const auto* dot_path = "mlp_loss_graph.dot";
                scalar_graph_viz::dump_dot(loss, dot_path);
                std::cout << "Wrote loss computation graph: " << dot_path << '\n';
                const int rc = std::system("dot -Tpng mlp_loss_graph.dot -o mlp_loss_graph.png");
                if (rc == 0) {
                    std::cout << "Rendered graph image: mlp_loss_graph.png\n";
                } else {
                    std::cout << "Failed to run Graphviz command. Install graphviz and run:\n";
                    std::cout << "dot -Tpng mlp_loss_graph.dot -o mlp_loss_graph.png\n";
                }
                loss_graph_dumped = true;
            }

            net.step(lr);

            total_loss += loss.value();
        }

        if (step % 100 == 0) {
            std::cout << "step=" << step << ", loss=" << total_loss << '\n';
        }
    }

    float mae = 0.f;
    std::cout << "\nPredictions after training:\n";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        std::vector<scalar> input;
        input.reserve(2);
        input.emplace_back(xs[i][0], std::vector<scalar*>{}, "x1");
        input.emplace_back(xs[i][1], std::vector<scalar*>{}, "x2");
        const auto& pred = net.forward_one(input);
        mae += std::fabs(pred.value() - ys[i]);
        std::cout << "[" << xs[i][0] << ", " << xs[i][1] << "] -> " << pred.value()
                  << " (target=" << ys[i] << ")\n";
    }
    std::cout << "MAE: " << (mae / static_cast<float>(xs.size())) << '\n';

    return 0;
}

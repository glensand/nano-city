#include "grad.h"
#include <cstdlib>
#include <iostream>

int main() {
    scalar x(2, {}, "x");
    scalar y(6, {}, "y");
    auto res = x * 2 + y.pow(10) + 10;

    // Start reverse pass from the result node.
    res.set_grad(1.f);
    res.backward();

    const auto* dot_path = "graph.dot";
    res.dump_dot(dot_path);
    std::cout << "Wrote computation graph: " << dot_path << '\n';
    const int rc = std::system("dot -Tpng graph.dot -o graph.png");
    if (rc == 0) {
        std::cout << "Rendered graph image: graph.png\n";
    } else {
        std::cout << "Failed to run Graphviz command. Install graphviz and run:\n";
        std::cout << "dot -Tpng graph.dot -ograph.png\n";
    }
}
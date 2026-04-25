#include "grad.h"
#include <iostream>

int main() {
    scalar x(1.5f, {}, "x");
    scalar y(-3.f, {}, "y");

    auto out = (x + 2.f) * y.relu() + x.pow(3.f);
    out.set_grad(1.f);
    out.backward();

    std::cout << "Forward output: " << out.value() << '\n';
    std::cout << "d(out)/d(x): " << x.grad() << '\n';
    std::cout << "d(out)/d(y): " << y.grad() << '\n';
    return 0;
}

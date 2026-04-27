#include "grad_scalar.h"
#include <cmath>
#include <iostream>

namespace {
float f(float x, float y) {
    scalar::scalar sx(x, {}, "x");
    scalar::scalar sy(y, {}, "y");
    auto out = sx * sy + sx.pow(3.f) + sy.relu();
    return out.value();
}

bool near(float lhs, float rhs, float tol = 1e-3f) {
    return std::fabs(lhs - rhs) <= tol;
}
} // namespace

int main() {
    constexpr float x0 = 1.2f;
    constexpr float y0 = -0.7f;
    constexpr float eps = 1e-3f;

    scalar::scalar x(x0, {}, "x");
    scalar::scalar y(y0, {}, "y");
    auto out = x * y + x.pow(3.f) + y.relu();
    out.set_grad(1.f);
    out.backward();

    const float dfdx_num = (f(x0 + eps, y0) - f(x0 - eps, y0)) / (2.f * eps);
    const float dfdy_num = (f(x0, y0 + eps) - f(x0, y0 - eps)) / (2.f * eps);

    const bool x_ok = near(x.grad(), dfdx_num);
    const bool y_ok = near(y.grad(), dfdy_num);

    std::cout << "Gradient validation for out = x*y + x^3 + relu(y)\n";
    std::cout << "x: autograd=" << x.grad() << ", numeric=" << dfdx_num
              << ", diff=" << std::fabs(x.grad() - dfdx_num)
              << ", status=" << (x_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "y: autograd=" << y.grad() << ", numeric=" << dfdy_num
              << ", diff=" << std::fabs(y.grad() - dfdy_num)
              << ", status=" << (y_ok ? "PASS" : "FAIL") << '\n';

    if (!(x_ok && y_ok)) {
        std::cerr << "Gradient validation failed\n";
        return 1;
    }

    std::cout << "All gradient checks passed\n";
    return 0;
}

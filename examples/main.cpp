#include "grad.h"

int main() {
    var x(0, {}, "x");
    var y(6, {}, "y");;
    auto res = x * 2 + y.pow(10) + 10;
}
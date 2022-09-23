#pragma once

#include <cmath>
#include <functional>

namespace neural_net {
    struct activation_function {
        static inline double linear(double x) {
            return x > 0 ? 1 : 0;
        }

        static inline double relu(double x) {
            return x > 0 ? x : 0;
        }

        static inline double sigmoid(double x) {
            return 1 / (1 + exp(-x));
        }

        static inline double tanh(double x) { // hyperbolic_tangent
            auto tmp = exp(2 * x);
            return ((tmp - 1) / (tmp + 1));
        }
    };

    using activation_function_t = std::function<double(double)>;
}
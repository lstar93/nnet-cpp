#pragma once

#include <cmath>
#include <functional>

namespace neural_net {
    struct cost_function {
        static inline double quadratic_cost(double returned_val, double expected_val) noexcept {
            auto tmp = returned_val - expected_val;
            return tmp * tmp;
        }

        static inline double quadratic_cost_derivative(double returned_val, double expected_val) noexcept {
            return 2 * (returned_val - expected_val);
        }
    };

    using cost_function_t = std::function<double(double, double)>;
}
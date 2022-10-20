#pragma once

#include <cmath>
#include <functional>

namespace neural_net {
    struct activation_function_t {
        virtual double activation(double x) = 0;
        virtual double derivative(double x) = 0;
        virtual ~activation_function_t() {};
    };

    struct linear_t: activation_function_t {
        double activation(double x) {
            return x > 0 ? 1 : 0;
        }

        double derivative(double x) {
            return x > 0 ? 1 : 0;
        }
    };

    struct relu_t: activation_function_t {
        double activation(double x) {
            return x > 0 ? x : 0;
        }

        double derivative(double x) {
            return x > 0 ? 1 : 0;
        }
    };

    struct sigmoid_t: activation_function_t {
        double activation(double x) {
            return 1 / (1 + exp(-x));
        }

        double derivative(double x) {
            auto tmp = 1 / (1 + exp(-x));
            return tmp * (1 - tmp);
        }
    };

    struct tanh_t: activation_function_t {
        double activation(double x) {
            auto tmp = exp(2 * x);
            return ((tmp - 1) / (tmp + 1));
        }

        double derivative(double x) {
            auto tmp = exp(2 * x);
            return ((tmp - 1) / (tmp + 1));
        }
    };

    namespace activation_function {
        static std::shared_ptr<linear_t> linear = std::make_shared<linear_t>();
        static std::shared_ptr<relu_t> relu = std::make_shared<relu_t>();
        static std::shared_ptr<sigmoid_t> sigmoid = std::make_shared<sigmoid_t>();
        static std::shared_ptr<tanh_t> tanh = std::make_shared<tanh_t>();
    }
}
#pragma once

#include <random>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <functional>

#include "layer.h"
#include "activation.h"
#include "cost.h"
#include "data.h"

namespace neural_net {

class neural_net_t {

    std::vector<layer_t> layers_vec;
    double net_cost = 1;

    public:

    constexpr const std::vector<layer_t>& layers() const noexcept {
        return layers_vec;
    }

    constexpr const layer_t& layer(size_t pos) const {
        return layers_vec.at(pos);
    }

    constexpr size_t size() const noexcept {
        return layers_vec.size();
    }

    neural_net_t& add_layer(size_t out_size, activation_function_t activ_func = activation_function::linear);

    std::vector<double> compute_output(data_chunk_t data_point);

    // compute neural network cost for single data point
    double cost(data_chunk_t& data_point);

    // compute neural network cost for several data points
    double cost(std::vector<data_chunk_t>& data_points);

    // get current network cost value
    constexpr double current_cost() const {
        return net_cost;
    }

    uint64_t classify(data_chunk_t& data_point);

    // apply gradients to all layers
    void apply_gradients(double learn_rate);

    // run a single iteration of gradient descent
    void learn(std::vector<data_chunk_t>& training_input_data, double learn_rate);
    void learn(data_chunk_t& training_input_data, double learn_rate);
};

}
#pragma once

#include <random>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <functional>

#include "activation.h"
#include "cost.h"
#include "data.h"

namespace neural_net {

using neuron_id_t = int;

struct neuron_t {
    neuron_id_t id;
    double value;
    double activation;

    neuron_t(): value(0), activation(0) {
        static neuron_id_t tmp_id = 0;
        id = tmp_id++;
    }

    constexpr neuron_t(const neuron_t& old) : 
        id(old.id), value(old.value){}

    neuron_t& operator=(const neuron_t& old) {
        id = old.id;
        value = old.value;
    }
};

class layer_t {
    std::vector<neuron_t> neurons_vec;
    size_t num_nodes_in, num_nodes_out;
    std::vector<double> biases;
    std::vector<double> cost_gradient_biases;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> cost_gradient_weights;
    std::shared_ptr<activation_function_t> activation_func;

    std::vector<double> inputs;

    public:

    layer_t(size_t _num_nodes_in, size_t _num_nodes_out, std::shared_ptr<activation_function_t> _activation_func = std::make_shared<linear_t>()) 
        : neurons_vec(_num_nodes_out),
          num_nodes_in(_num_nodes_in), num_nodes_out(_num_nodes_out),
          biases(_num_nodes_out), cost_gradient_biases(_num_nodes_out),
          weights(_num_nodes_in, std::vector<double>(_num_nodes_out, 1)), // init weights with 1
          cost_gradient_weights(_num_nodes_in, std::vector<double>(_num_nodes_out, 0)), // init weights gradients with 0
          activation_func(_activation_func), inputs(_num_nodes_in) {}

    constexpr const std::vector<neuron_t>& neurons() const noexcept {
        return neurons_vec;
    }

    constexpr std::vector<neuron_t>& neurons() noexcept {
        return neurons_vec;
    }

    constexpr size_t input_size() const {
        return num_nodes_in;
    }

    constexpr size_t output_size() const {
        return num_nodes_out;
    }

    constexpr size_t size() const noexcept {
        return neurons_vec.size();
    }

    constexpr neuron_t& at(size_t pos) {
        return neurons_vec.at(pos);
    }

    constexpr double& weight(size_t i, size_t j) {
        return weights[i][j];
    }

    constexpr double& cost_gradient_weight(size_t i, size_t j) {
        return cost_gradient_weights[i][j];
    }

    constexpr double& bias(size_t i) {
        return biases[i];
    }

    constexpr double& cost_gradient_bias(size_t i) {
        return cost_gradient_biases[i];
    }

    void init_weights();

    neuron_t neuron(neuron_id_t id) const;

    std::vector<double> compute_output(data_chunk_t& data_point);

    std::vector<double> compute_output_layer_node_values(std::vector<double>& expected_outputs);

    std::vector<double> compute_hidden_layer_node_values(layer_t& next_layer, std::vector<double>& next_node_values);

    // update the weights and biases based on the cost gradients (gradient descent)
    void apply_gradients(double learn_rate);

    void update_gradients(std::vector<double>& node_values);

    void clear_gradients();
};
}
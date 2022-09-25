#include "nnet.h"

namespace neural_net {

std::chrono::high_resolution_clock::time_point tictoc::start = std::chrono::high_resolution_clock::now();

// neural_net_t

neural_net_t& neural_net_t::add_layer(size_t out_size, std::shared_ptr<activation_function_t> activ_func) {
    size_t input_size = (layers_vec.size() > 0 ? layers_vec.at(layers_vec.size()-1).size() : out_size);
    layer_t layer {input_size, out_size, activ_func};
    layer.init_weights();
    layers_vec.push_back(std::move(layer));
    
    return *this;
}

std::vector<double> neural_net_t::compute_output(data_chunk_t data_point) {
    for(auto i=0; i<layers_vec.size(); ++i) { // compute output of each layer
        data_point.input = layers_vec[i].compute_output(data_point);
    }
    return data_point.input;
}

uint64_t neural_net_t::classify(data_chunk_t& data_point) {
    auto output = compute_output(data_point);
    auto max = std::max_element(output.begin(), output.end());
    return std::distance(output.begin(), max);
}

double neural_net_t::cost(data_chunk_t& data_point) {
    auto outputs = compute_output(data_point);
    double cost = 0;

    for(size_t i=0; i<outputs.size(); ++i) {
        cost += cost_function::quadratic_cost(outputs[i], data_point.expected_output[i]);
    }

    return cost;
}

double neural_net_t::cost(std::vector<data_chunk_t>& data_points) {
    double total_cost = 0;
    for(auto& dp: data_points) {
        total_cost += cost(dp);
    }
    return (total_cost / data_points.size());
}

void neural_net_t::apply_gradients(double learn_rate) {
    for(auto i=0; i<layers_vec.size(); ++i) {
        layers_vec[i].apply_gradients(learn_rate);
    }
}

void neural_net_t::update_gradients(data_chunk_t& data_point) {
    // Run the inputs trough the network. During this process, each layer will,
    // store the values we need, such as weighted inputs and activations
    auto outputs = compute_output(data_point);

    auto& out_layer = layers_vec.at(layers_vec.size() - 1);
    auto node_values = out_layer.compute_output_layer_node_values(data_point.expected_output);
    out_layer.update_gradients(node_values);

    // Update gradients of the hidden layer -> back propagation algorithm
    for(int hidden_layer_idx = layers_vec.size() - 2; hidden_layer_idx >= 0; --hidden_layer_idx) {
        size_t siz = hidden_layer_idx;
        auto& hidden_layer = layers_vec[hidden_layer_idx]; 
        node_values = hidden_layer.compute_hidden_layer_node_values(layers_vec[hidden_layer_idx + 1], node_values);
        hidden_layer.update_gradients(node_values);
    }
}

void neural_net_t::clear_gradients() {
    for(auto& layer: layers_vec) {
        layer.clear_gradients();
    }
}

void neural_net_t::learn(std::vector<data_chunk_t>& training_input_data, double learn_rate) {
    // Update gradients using back propagation method
    for(auto& dp: training_input_data) {
        update_gradients(dp);
    }

    net_cost = cost(training_input_data);

    std::cout << "cost: " << net_cost << std::endl;

    // Gradient descent step: update all the weights and biases in the network
    apply_gradients(learn_rate / training_input_data.size());

    // Reset all gradients to zero to be create for the next training batch
    clear_gradients();
}

}
#include "layer.h"

namespace neural_net {

// layer_t

void layer_t::init_weights() {
    std::uniform_real_distribution<double> unif(-1,1);
    std::default_random_engine re;
    for(auto& col: weights) {
        for(auto& row: col) {
            row = unif(re);
        }
    }
}

neuron_t layer_t::neuron(neuron_id_t id) const {
    auto neuron = std::find_if(neurons_vec.begin(), neurons_vec.end(), [&](auto& n) { return id == n.id; });
    if(neuron != neurons_vec.end()) {
        return *neuron;
    }
    throw std::out_of_range("No neuron with id " + std::to_string(id) + " found!");
}

std::vector<double> layer_t::compute_output(data_chunk_t& data_point) {
    std::vector<double> output;
    for(auto i=0; i<num_nodes_out; ++i) {
        neurons_vec[i].value = biases[i];
        for(auto j=0; j<num_nodes_in; ++j) {
            neurons_vec[i].value += data_point.input[j] * weights[j][i];
        }
        output.push_back(activation_func(neurons_vec[i].value));
    }
    return output;
}

void layer_t::apply_gradients(double learn_rate) {
    for(auto i=0; i<num_nodes_out; ++i) {
        biases[i] -= cost_gradient_biases[i] * learn_rate;
        for(auto j=0; j<num_nodes_in; ++j) {
            cost_gradient_weights[j][i] -= cost_gradient_weights[j][i] * learn_rate;
        }
    } 
}

}
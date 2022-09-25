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
    std::vector<double> activations;

    // comute neuron output value
    for(auto i=0; i<num_nodes_out; ++i) {
        neurons_vec[i].value = biases[i];
        for(auto j=0; j<num_nodes_in; ++j) {
            neurons_vec[i].value += data_point.input[j] * weights[j][i];
        }
    }

    // compute activation value (output value passed trough activation function)
    for(auto i=0; i<num_nodes_out; ++i) {
        neurons_vec[i].activation = activation_func->activation(neurons_vec[i].value);
        activations.push_back(neurons_vec[i].activation);
    }

    inputs = data_point.input;

    return activations;
}

void layer_t::apply_gradients(double learn_rate) {
    for(auto i=0; i<num_nodes_out; ++i) {
        biases[i] -= cost_gradient_biases[i] * learn_rate;
        for(auto j=0; j<num_nodes_in; ++j) {
            cost_gradient_weights[j][i] -= cost_gradient_weights[j][i] * learn_rate;
        }
    } 
}

void layer_t::update_gradients(std::vector<double>& node_values) {
    for(auto i=0; i<num_nodes_out; ++i) {
        double n_val = node_values[i];
        for(auto j=0; j<num_nodes_in; ++j) {
            // Evalueate the partial derivative: cost / weight of current connection
            double derivative_cost_weight = inputs[j] * n_val;

            // cost gradient weights array stores there partial derivatives for each weight.
            // the derivative is being added to the array here because ultimately we want
            // to calculate the average gradient across all the data in the training batch
            cost_gradient_weights[j][i] += derivative_cost_weight;
        }

        // Evalueate the partial derivatives: cost / bias of current node
        double derivative_cost_bias = node_values[i];
        cost_gradient_biases[i] += derivative_cost_bias;
    } 
}

std::vector<double> layer_t::compute_output_layer_node_values(std::vector<double>& expected_outputs) {
    std::vector<double> node_values;
    auto neurons_iter = neurons_vec.begin();
    for(auto i=0; i<num_nodes_out; ++i) {
        double cost_derivative = cost_function::quadratic_cost_derivative(neurons_iter->activation, expected_outputs[i]);
        double activation_derivative = activation_func->derivative(neurons_iter->value);
        node_values.push_back(activation_derivative * cost_derivative);
        neurons_iter++;
    }

    return node_values;
}

std::vector<double> layer_t::compute_hidden_layer_node_values(layer_t& next_layer, std::vector<double>& next_node_values) {
    std::vector<double> new_node_values;
    
    auto neurons_iter = neurons_vec.begin();
    for(auto i=0; i<num_nodes_out; ++i) { 
        double new_node_value = 0;
        for(auto j=0; j< next_layer.size(); ++j) {
            // partial derivative of the weighted input with respect to the input
            double weighted_input_derivative = next_layer.weights[i][j];
            new_node_value += weighted_input_derivative * next_node_values[j];
        }
        new_node_value *= activation_func->derivative(neurons_iter->value);
        new_node_values.push_back(new_node_value);
        neurons_iter++;
    }   

    return new_node_values;
}

void layer_t::clear_gradients() {
    for(auto i=0; i<cost_gradient_biases.size(); ++i) {
        cost_gradient_biases[i] = 0;
    }

    for(auto i=0; i<cost_gradient_weights.size(); ++i) {
        for(auto j=0; j<cost_gradient_weights[i].size(); ++j) {
            cost_gradient_weights[i][j] = 0;
        }
    }
}

}
#include "nnet.h"

namespace neural_net {

std::chrono::high_resolution_clock::time_point tictoc::start = std::chrono::high_resolution_clock::now();

// neural_net_t

neural_net_t& neural_net_t::add_layer(size_t out_size, activation_function_t activ_func) {
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

void neural_net_t::learn(std::vector<data_chunk_t>& training_input_data, double learn_rate) {
    const double h = 0.0001;
    net_cost = cost(training_input_data);

    for(auto& layer: layers_vec) {
        // calculate the cost gradient for the current weights
        tictoc::tic();
        for(auto i=0; i<layer.input_size(); ++i) {
            for(auto j=0; j<layer.output_size(); ++j) {
                layer.weight(i,j) += h;
                double delta_cost = cost(training_input_data) - net_cost;
                layer.weight(i,j) -= h;
                layer.cost_gradient_weight(i,j) = delta_cost / h;
            }
        }
        std::cout << "1. elapsed s: " << tictoc::toc() << std::endl;
        tictoc::tic();
        // calculate the cost gradient for the current bias
        for(auto i=0; i<layer.output_size(); ++i) {
            layer.bias(i) += h;
            double delta_cost = cost(training_input_data) - net_cost;
            layer.bias(i) -= h;
            layer.cost_gradient_bias(i) = delta_cost / h;
        }
    }
    std::cout << "2. elapsed s: " << tictoc::toc() << std::endl;

    // apply gardients to all layers
    tictoc::tic();
    apply_gradients(learn_rate);
    std::cout << "3. elapsed s: " << tictoc::toc() << std::endl;
}

void neural_net_t::learn(data_chunk_t& training_input_data, double learn_rate) {
    const double h = 0.0001;
    net_cost = cost(training_input_data);

    for(auto& layer: layers_vec) {
        // calculate the cost gradient for the current weights
        for(auto i=0; i<layer.input_size(); ++i) {
            for(auto j=0; j<layer.output_size(); ++j) {
                layer.weight(i,j) += h;
                double delta_cost = cost(training_input_data) - net_cost;
                layer.weight(i,j) -= h;
                layer.cost_gradient_weight(i,j) = delta_cost / h;
            }
        }

        // calculate the cost gradient for the current bias
        for(auto i=0; i<layer.output_size(); ++i) {
            layer.bias(i) += h;
            double delta_cost = cost(training_input_data) - net_cost;
            layer.bias(i) -= h;
            layer.cost_gradient_bias(i) = delta_cost / h;
        }
    }

    // apply gardients to all layers
    apply_gradients(learn_rate);
}

}
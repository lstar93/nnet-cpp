#define _USE_MATH_DEFINES // for sin/log

#include <random>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <unordered_map>

#include "plot.h"
#include "nnet.h"
#include "data.h"

namespace plt = matplotlibcpp;

std::vector<neural_net::data_chunk_t> generate_learn_data(size_t size) {
    std::uniform_real_distribution<double> unif(0,200);
    std::default_random_engine re;
    std::vector<neural_net::data_chunk_t> ret;
    for(size_t i=0; i<size; ++i) {
        std::vector<double> data {unif(re), unif(re)};
        neural_net::data_chunk_t chunk;
        chunk.input = data;
        if(data.at(0) < 10 && data.at(1) < 10) {
            chunk.expected_output = {1, 0};
        }
        else {
            chunk.expected_output = {0, 1};
        }
        ret.push_back(std::move(chunk));
    }

    return ret;
}

int main() {

    neural_net::neural_net_t nnet;
    nnet.add_layer(2)
        .add_layer(64, neural_net::activation_function::relu)
        .add_layer(128, neural_net::activation_function::relu)
        .add_layer(256, neural_net::activation_function::relu)
        .add_layer(128, neural_net::activation_function::relu)
        .add_layer(64, neural_net::activation_function::relu)
        .add_layer(2, neural_net::activation_function::relu);

    // test
    size_t batch_size = 32;
    size_t chunk_start = 0;
    size_t chunk_end = batch_size;

    auto data_chunks = generate_learn_data(2048);
    auto dit = data_chunks.begin();

    while(nnet.current_cost() > 0.01) {
        //d.dump();
        std::vector<neural_net::data_chunk_t> batch(data_chunks.begin()+chunk_start, data_chunks.begin()+chunk_end);
        chunk_start = chunk_end;
        chunk_end += batch_size;
        if(chunk_end >= data_chunks.size()) {
            chunk_start = 0;
            chunk_end = batch_size;
        }

        nnet.learn(batch, 0.05);
    }

    neural_net::data_chunk_t test;
    test.input = {3, 2};
    std::cout << nnet.classify(test) << std::endl; // 1

    test.input = {2, 20};
    std::cout << nnet.classify(test) << std::endl; // 1

    test.input = {10, 10};
    std::cout << nnet.classify(test) << std::endl; // 0

    test.input = {15, 12};
    std::cout << nnet.classify(test) << std::endl; // 0

    test.input = {112, 35};
    std::cout << nnet.classify(test) << std::endl; // 0

    test.input = {1, 33};
    std::cout << nnet.classify(test) << std::endl; // 1

    test.input = {22, 25};
    std::cout << nnet.classify(test) << std::endl; // 0

    return 0;
}
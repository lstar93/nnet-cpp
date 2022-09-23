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

std::vector<std::vector<double>> generate_data(size_t size) {
    std::uniform_real_distribution<double> unif(-10,10);
    std::default_random_engine re;
    std::vector<std::vector<double>> ret;
    for(size_t i=0; i<size; ++i) {
        ret.emplace_back(std::vector<double>{unif(re), unif(re)});
    }

    return ret;
}

std::vector<neural_net::data_chunk_t> generate_learn_data(size_t size) {
    std::uniform_real_distribution<double> unif(0,7);
    std::default_random_engine re;
    std::vector<neural_net::data_chunk_t> ret;
    for(size_t i=0; i<size; ++i) {
        std::vector<double> data {unif(re), unif(re)};
        neural_net::data_chunk_t chunk;
        chunk.input = data;
        if(data.at(0) < 5 && data.at(1) < 4) {
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
        .add_layer(64, neural_net::activation_function::relu)
        .add_layer(2, neural_net::activation_function::relu);

    // test
    size_t batch_size = 32;
    size_t chunk_start = 0;
    size_t chunk_end = batch_size;

    auto data_chunks = generate_learn_data(128);
    auto dit = data_chunks.begin();
    while(nnet.current_cost() > 0.1) {
        //d.dump();
        std::vector<neural_net::data_chunk_t> batch(data_chunks.begin()+chunk_start, data_chunks.begin()+chunk_end);
        chunk_start = chunk_end;
        chunk_end += batch_size;
        if(chunk_end >= data_chunks.size()) {
            chunk_start = 0;
            chunk_end = batch_size;
        }
        std::cout << "chunk_start: "<< chunk_start << ", chunk_end: " << chunk_end << std::endl;

        nnet.learn(batch, 0.01);
        std::cout << nnet.current_cost() << std::endl;
    }

    neural_net::data_chunk_t test;
    test.input = {3, 3};
    std::cout << nnet.classify(test) << std::endl;

    test.input = {3, 7};
    std::cout << nnet.classify(test) << std::endl;

    test.input = {6, 2};
    std::cout << nnet.classify(test) << std::endl;

    test.input = {1, 3.5};
    std::cout << nnet.classify(test) << std::endl;  
    //std::cout << nnet.size() << std::endl;

    /*size_t inSize = 2, outSize = 2;

    auto snnInputs = init_simple_neural_network(inSize, outSize);

    auto points = plotter::get_random_points({0,30}, {0,30}, 50);

    auto myplt = plotter::plotter::create(plotter::plottype::joint);

    //while(true) {
    double weight11 = 0.0;
    double weight12 = 0.0;
    double bias1 = 0.0;

    double weight21 = 0.0;
    double weight22 = 0.0;
    double bias2 = 0.0;

    std::cout << "set weight11: ";
    std::cin >> weight11;

    std::cout << "set weight12: ";
    std::cin >> weight12;

    std::cout << "set weight21: ";
    std::cin >> weight21;

    std::cout << "set weight22: ";
    std::cin >> weight22;

    std::cout << "set bias1: ";
    std::cin >> bias1;

    std::cout << "set bias2: ";
    std::cin >> bias2;

        // set weight11: 1
        // set weight12: 0.9
        // set weight21: 0.7
        // set weight22: 0.8

    std::vector<plotter::point_t> points1;
    std::vector<plotter::point_t> points2;
    // classify
    for(auto& p: points) {
        snnInputs[0].value = p.x;
        snnInputs[0].weights[0] = weight11;
        snnInputs[0].weights[1] = weight12;
        // snnInputs[0].bias = bias1;

        snnInputs[1].value = p.y;
        snnInputs[1].weights[0] = weight21;
        snnInputs[1].weights[1] = weight22;
        // snnInputs[1].bias = bias2;

        auto snnOutputs = compute_outputs(snnInputs, outSize);

        if(classify(p.x, weight11, weight12, p.y, weight21, weight22, bias1, bias2)) {
        // if(snnOutputs[0].value < snnOutputs[1].value) {
            points1.push_back(p);
        } else {
            points2.push_back(p);
        }
    }

    std::cout << "first group size = " << points1.size() << std::endl;
    std::cout << "second group size = " << points2.size() << std::endl;

    myplt.plot_points({points1, points2});

    plotter::plotter::show();*/
    //}

    return 0;
}
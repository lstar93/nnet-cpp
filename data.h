#pragma once

#include <vector>
#include <iostream>
#include "helpers.h"

namespace neural_net {

struct data_chunk_t {
    std::vector<double> input;
    std::vector<double> expected_output;

    void dump() {
        std::cout << "input: ";
        print_container(input);
        std::cout << ", expected output: ";
        print_container(expected_output);
        std::cout << std::endl;
    }
};

}
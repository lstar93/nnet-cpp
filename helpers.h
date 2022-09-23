#pragma once

#include <vector>
#include <iostream>
#include <chrono>

namespace neural_net {

struct tictoc {
    static std::chrono::high_resolution_clock::time_point start;
    static void tic() {
        start = std::chrono::high_resolution_clock::now();
    }
    static float toc() {
        auto f_secs = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start);
        return f_secs.count();
    }
};

// template<typename T, template <typename> class VEC>
template<typename T>
void print_container(T& vec) {
    std::cout << "{ ";
    for(auto& elem: vec) {
        std::cout << elem << ((elem != *(vec.end()-1)) ? ", " : "");
    }
    std::cout << " }";
}

struct print_verbose {
    static const bool verbose = true;

    static inline void print(const std::string& msg) {
        if(verbose) {
            std::cout << msg;
        }
    }

    static inline void println(const std::string& msg) {
        if(verbose) {
            std::cout << msg << "\n";
        }
    }
};

}
#include "tester/rand_gen.hpp"
#include "tester/opt_test_case.cuh"
#include <iostream>


int main(int argc, char** argv) {
    try {
        std::vector<u32> keys;
        rand_gen(keys, 1_M);
        store(keys, "../data/");


    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
} 
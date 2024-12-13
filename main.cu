#include "tester/rand_gen.hpp"
#include "tester/opt_test_case.cuh"
#include <iostream>


int main(int argc, char** argv) {
    try {
        std::vector<u32> keys;
        load(keys, "../data/task1/rand_keys_2^24.bin");
        OptTestCase test_case(10, 32_M);
        test_case.run(keys, keys, std::vector<std::size_t>{90, 120, 150});
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
} 
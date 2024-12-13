#include "tester/opt_test_case.cuh"
#include "tester/rand_gen.hpp"
#include <vector>
#include <array>
#include <cmath>

int main() {
    std::vector<u32> keys;
    load(keys, "../data/task2/rand_keys_set0_2^24.bin");

    OptTestCase test (10, 1.4 * 16_M);
    test.run(
        keys, 
        {(const uint32_t *)nullptr, {0UL}}, 
        std::array<std::size_t, 10>{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
    );
    return 0;
}
#include "tester/opt_test_case.cuh"
#include "tester/rand_gen.hpp"
#include <vector>
#include <array>
#include <cmath>

int main() {
    std::vector<u32> keys;
    load(keys, "../data/task2/rand_keys_set0_2^24.bin");
    std::size_t table_size_base = 16_M;
    std::array<float, 13> ratios = {1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
    std::size_t table_size;
    std::size_t evict_chain_len;

    for(auto ratio : ratios) {
        table_size = static_cast<std::size_t>(std::ceil(table_size_base * ratio));
        evict_chain_len = static_cast<std::size_t>(4 * std::log2(table_size));
        OptTestCase test (10, table_size);
        test.run(keys,{(const uint32_t *)nullptr, {0UL}}, std::array<std::size_t, 1>{evict_chain_len});
    }
    return 0;
}
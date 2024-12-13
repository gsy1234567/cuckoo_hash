#include "tester/test_case.hpp"
#include "tester/rand_gen.hpp"
#include <iostream>
#include <vector>


int main() {
    try {
        SingleTestCase test(4, 32_MB);
        test.init();
        for(auto s = 10_u64 ; s != 25_u64 ; ++s) {
            std::vector<u32> keys;
            char buf[512];
            sprintf(buf, "../data/task1/rand_keys_2^%lu.bin", s);
            load(keys, buf);
            sprintf(buf, "Insert 2^%lu keys into 32M hash table\n", s);
            std::cout << buf; 
            test.run(keys, std::span<const u32>());
            std::cout << std::endl;
            test.clear();
        }
        test.deinit();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
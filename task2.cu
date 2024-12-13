#include "tester/test_case.hpp"
#include "tester/rand_gen.hpp"
#include <iostream>
#include <vector>

void make_search_set(std::vector<u32>& dst, std::span<const u32> set0, std::span<const u32> set1, float ratio);

int main() {
    try {
        SingleTestCase test(4, 32_MB);
        std::vector<u32> keys0;
        std::vector<u32> keys1;
        std::vector<u32> mix_keys;
        char buf[512];

        test.init();
        load(keys0, "../data/task2/rand_keys_set0_2^24.bin");
        load(keys1, "../data/task2/rand_keys_set1_2^24.bin");
        for(int i = 0 ; i <= 100 ; i += 10) {
            float ratio = i / 100.f;
            make_search_set(mix_keys, keys0, keys1, ratio);
            sprintf(buf, "Insert 2^24 keys into 32M hash table\nSearch 2^24 keys with %d percent keys exists\n", 100-i);
            std::cout << buf;
            test.run(keys0, mix_keys);
            test.clear();
            std::cout << std::endl;
        }
        test.deinit();
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

void make_search_set(std::vector<u32>& dst, std::span<const u32> set0, std::span<const u32> set1, float ratio) {
    assert(set0.size() == 16_MB);
    assert(set1.size() == 16_MB);

    dst.resize(16_MB);
    std::size_t n0, n1;
    n1 = static_cast<u64>(16_MB * ratio);
    n0 = 16_MB - n1;
    std::copy_n(set0.begin(),  n0, dst.begin());
    std::copy_n(set1.begin(), n1, dst.begin() + n0);
    assert(dst.size() == 16_MB);
}
#pragma once

#include "../include/cuckoo.cuh"
#include <span>

class SingleTestCase {
    protected:
        using cuckoo2_t = cuckoo_hash_table<u32, 2>;
        using cuckoo3_t = cuckoo_hash_table<u32, 3>;
        std::size_t m_test_times;
        std::size_t m_table_size;
        cuckoo2_t m_cuckoo2;
        cuckoo3_t m_cuckoo3;
    public:
        SingleTestCase(std::size_t test_times, std::size_t table_size) : m_test_times(test_times), m_table_size(table_size) {}

        void init();

        void clear();

        void deinit();

        void run(std::span<const u32> insert_keys, std::span<const u32> search_keys);
    
};
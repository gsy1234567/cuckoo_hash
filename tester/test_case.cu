#include "test_case.hpp"
#include <chrono>
#include <vector>
#include <unordered_set>
#include <iostream>

void SingleTestCase::init() {
    cuckoo2_t::init(m_cuckoo2, m_table_size);
    cuckoo3_t::init(m_cuckoo3, m_table_size);
}

void SingleTestCase::clear() {
    cuckoo2_t::clear(m_cuckoo2);
    cuckoo3_t::clear(m_cuckoo3);
}

void SingleTestCase::deinit() {
    cuckoo2_t::deinit(m_cuckoo2);
    cuckoo3_t::deinit(m_cuckoo3);
}

void SingleTestCase::run(std::span<const u32> insert_keys, std::span<const u32> search_keys) {
    std::chrono::nanoseconds insert2_times {0};
    std::chrono::nanoseconds search2_times {0};
    std::chrono::nanoseconds insert3_times {0};
    std::chrono::nanoseconds search3_times {0};
    bool* search_results = new bool[search_keys.size()];
    //std::unordered_set<u32> inserted {insert_keys.begin(), insert_keys.end()};
    //std::size_t error_times;

    for(auto test_idx = 0_sz ; test_idx != m_test_times ; ++test_idx) {
        insert2_times += cuckoo2_t::insert(m_cuckoo2, insert_keys.data(), insert_keys.size());
        if(!search_keys.empty()) {
            search2_times += cuckoo2_t::search(m_cuckoo2, search_keys.data(), search_results, search_keys.size());
        }
        cuckoo2_t::clear(m_cuckoo2);

        // error_times = 0;
        // for(auto search_idx = 0_sz ; search_idx != search_keys.size() ; ++search_idx) {
        //     if((inserted.find(search_keys[search_idx]) != inserted.end()) != search_results[search_idx]) {
        //         ++error_times;
        //     }
        // }
        // if(error_times != 0) {
        //     std::cerr << "error times: " << error_times << std::endl;
        // }

        insert3_times += cuckoo3_t::insert(m_cuckoo3, insert_keys.data(), insert_keys.size());
        if(!search_keys.empty()) {
            search3_times += cuckoo3_t::search(m_cuckoo3, search_keys.data(), search_results, search_keys.size());
        }
        cuckoo3_t::clear(m_cuckoo3);

        // error_times = 0;
        // for(auto search_idx = 0_sz ; search_idx != search_keys.size() ; ++search_idx) {
        //     if((inserted.find(search_keys[search_idx]) != inserted.end()) != search_results[search_idx]) {
        //         ++error_times;
        //     }
        // }
        // if(error_times != 0) {
        //     std::cerr << "error times: " << error_times << std::endl;
        // }
    }
    printf("insert2 bandwidth: %lf M/sec\n", (double)insert_keys.size() * 1e3 * m_test_times / insert2_times.count());
    printf("insert3 bandwidth: %lf M/sec\n", (double)insert_keys.size() * 1e3 * m_test_times / insert3_times.count());
    if(!search_keys.empty()) {
        printf("search2 bandwidth: %lf M/sec\n", (double)search_keys.size() * 1e3 * m_test_times / search2_times.count());
        printf("search3 bandwidth: %lf M/sec\n", (double)search_keys.size() * 1e3 * m_test_times / search3_times.count());
    }

    delete [] search_results;
}
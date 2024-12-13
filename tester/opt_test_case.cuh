#pragma once

#include "../include/dycuckoo.cuh"
#include "../include/utils.hpp"
#include <span>
#include <chrono>
#include <numeric>

class OptTestCase {
    protected:

        std::size_t m_test_times;
        std::size_t m_table_size;

        template<std::size_t TableNum, std::size_t GroupSize>
        void run_single(std::span<const u32> insert_keys, std::span<const u32> search_keys, std::span<const std::size_t> max_evicts) {
            using cuckoo_t = dycuckoo_hash_table<u32, TableNum, GroupSize>;

            typename cuckoo_t::host_error_table_t host_error_table;

            cuckoo_t ht;
            for(auto max_evict : max_evicts) {

                std::chrono::nanoseconds insert_time {0};
                std::chrono::nanoseconds search_time {0};

                printf("---Start Test---\n");
                printf("\ttable size = %lu\n", m_table_size);
                printf("\tevict chain length = %lu\n", max_evict);
                printf("\tinsert keys number = %lu\n", insert_keys.size());
                printf("\tsearch keys number = %lu\n", search_keys.size());
                printf("\ttest times = %lu\n", m_test_times);
                printf("\ttable number = %lu\n", TableNum);
                printf("\tgroup size = %lu\n", GroupSize);

                for(auto test_iter = 0_sz ; test_iter < m_test_times ; ++test_iter) {
                    cuckoo_t::init(ht, m_table_size, max_evict, 1_K);
                    std::default_random_engine egn;

                    std::size_t insert_num;
                    std::chrono::nanoseconds _insert_time;
                    bool insert_succeed = false;

                    for(insert_num = 0_sz ; insert_num < 30_sz ; ++insert_num) {
                        _insert_time = cuckoo_t::insert(ht, insert_keys.data(), insert_keys.size());
                        cuckoo_t::check_error(ht, host_error_table);
                        if(host_error_table.empty()) {
                            insert_time += _insert_time;
                            insert_succeed = true;
                            break;
                        } else {
                            cuckoo_t::clear(ht);
                            cuckoo_t::update(ht, egn);
                        }
                    }

                    if(!insert_succeed) {
                        printf("\tInsert failed more than 30 times!\n");
                        cuckoo_t::deinit(ht);
                        goto End;
                    }

                    insert_time += _insert_time;
                    if(!search_keys.empty()) {
                        std::vector<u32> search_results(search_keys.size() / std::numeric_limits<u32>::digits + 1);
                        search_time += cuckoo_t::search(ht, search_keys.data(), search_keys.size(), search_results.data(), search_results.size());
                    }

                    cuckoo_t::deinit(ht);
                }
                printf("\tinsert time = %f ms\n", insert_time.count() / (1e6f * m_test_times));
                printf("\tsearch time = %f ms\n", search_time.count() / (1e6f * m_test_times));
                printf("\tinsert bandwidth = %f M insertions/sec\n", insert_keys.size() * 1e3 * m_test_times / insert_time.count());
                printf("\tsearch bandwidth = %f M searchs/sec\n",  search_keys.empty() ? 0.0f : search_keys.size() * 1e3 * m_test_times / search_time.count());
                End:
                printf("---End Test---\n");
            }
        }
        
    public:
        OptTestCase(std::size_t test_times, std::size_t table_size) : m_test_times(test_times), m_table_size(table_size) {}

        void run(std::span<const u32> insert_keys, std::span<const u32> search_keys = std::span<const u32>(), std::span<const std::size_t> max_evicts = std::span<const std::size_t>());
};
#pragma once

#include "data_layout.cuh"
#include "utils.hpp"
#include "hash_helper.cuh"
#include <chrono>
#include <cmath>

template<std::unsigned_integral Key, std::size_t TableNum>
class cuckoo_hash_table;

namespace {

    template<std::unsigned_integral Key, std::size_t TableNum>
    __global__ void insert_kernel(const cuckoo_hash_table<Key, TableNum> table, const Key* keys, const std::size_t num);

    template<std::unsigned_integral Key, std::size_t TableNum>
    __global__ void search_kernel(const cuckoo_hash_table<Key, TableNum> table, const Key* keys, bool* result, const std::size_t num);
}


template<std::unsigned_integral Key, std::size_t TableNum>
class cuckoo_hash_table {
    public:
        using key_type = Key;
        using data_layout_t = DataLayout<key_type>;
        using single_table_t = data_layout_t::dev_single_table;

        static constexpr std::size_t table_num = TableNum;
        std::size_t m_table_size = 0;
        std::size_t m_max_evict_num = 0;
        single_table_t m_tables [table_num];

        cuckoo_hash_table() = default;

        static void init(cuckoo_hash_table& table, std::size_t table_size) {
            table.m_table_size = table_size;
            table.m_max_evict_num = 4_sz * std::log2(table_size);
            for(auto single_table_idx = 0_sz ; single_table_idx < table_num ; ++single_table_idx) {
                single_table_t::init(table.m_tables[single_table_idx], table_size);
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
        }
        
        static std::chrono::nanoseconds insert(cuckoo_hash_table& table, const key_type* keys, std::size_t num) {
            key_type* dev_keys = nullptr;
            cuda_timer timer;
            std::chrono::nanoseconds ns;

            timer.init();
            init_dev_data_async<key_type>(dev_keys, keys, num);
            CUDA_CHECK(cudaStreamSynchronize(0));

            timer.start(0);
            launch_cg_kernel((const void*)&insert_kernel<Key, TableNum>, 0, 0, table, dev_keys, num);
            timer.finish(0);

            CUDA_CHECK(cudaStreamSynchronize(0));
            ns = timer.duration();
            deinit_dev_data_async<key_type>(dev_keys);
            CUDA_CHECK(cudaStreamSynchronize(0));
            timer.deinit();

            return ns;
        }

        static std::chrono::nanoseconds search(cuckoo_hash_table& table, const key_type* keys, bool* res, std::size_t num) {
            key_type *dev_keys = nullptr;
            bool *dev_res = nullptr;
            cuda_timer timer;
            std::chrono::nanoseconds ns;

            timer.init();
            init_dev_data_async<key_type>(dev_keys, keys, num);
            init_dev_data_async<bool>(dev_res, nullptr, num);
            CUDA_CHECK(cudaStreamSynchronize(0));

            timer.start(0);
            launch_cg_kernel((const void*)&search_kernel<Key, TableNum>, 0, 0, table, dev_keys, dev_res, num);
            timer.finish(0);

            CUDA_CHECK(cudaStreamSynchronize(0));
            ns = timer.duration();
            CUDA_CHECK(cudaMemcpyAsync((void*)res, (const void*)dev_res, num * sizeof(bool), cudaMemcpyDeviceToHost));
            deinit_dev_data_async<key_type>(dev_keys);
            deinit_dev_data_async<bool>(dev_res);
            CUDA_CHECK(cudaStreamSynchronize(0));
            timer.deinit();

            return ns;
        }

        static void clear(cuckoo_hash_table& table) {
            for(auto single_table_idx = 0_sz ; single_table_idx < table.table_num; ++single_table_idx) {
                single_table_t::clear(table.m_tables[single_table_idx], table.m_table_size);
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
        }

        static void deinit(cuckoo_hash_table& table) {
            table.m_table_size = 0;
            for(auto single_table_idx = 0_sz ; single_table_idx < table_num ; ++single_table_idx) {
                single_table_t::deinit(table.m_tables[single_table_idx]);
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
        }

};

namespace {

    template<std::unsigned_integral Key, std::size_t TableNum>
    __global__ void insert_kernel(const cuckoo_hash_table<Key, TableNum> table, const Key* keys, const std::size_t num) {
        Key key;
        std::size_t pos;
        std::size_t idx;
        std::size_t iter;
        for(idx = blockDim.x * blockIdx.x + threadIdx.x ; idx < num ; idx += gridDim.x * blockDim.x) {
            key = keys[idx];
            for(iter = 0_sz ; iter != table.m_max_evict_num ; ++iter) {
                if constexpr (TableNum == 2) {
                    pos = hash_funcs::hash1<Key>(key) % table.m_table_size;
                    key = atomicExch(table.m_tables[0].keys + pos, key);
                    if(key == cuckoo_hash_table<Key, TableNum>::data_layout_t::empty_key) break;
                    pos = hash_funcs::hash2<Key>(key) % table.m_table_size;
                    key = atomicExch(table.m_tables[1].keys + pos, key);
                    if(key == cuckoo_hash_table<Key, TableNum>::data_layout_t::empty_key) break;
                } else if constexpr (TableNum == 3) {
                    pos = hash_funcs::hash1<Key>(key) % table.m_table_size;
                    key = atomicExch(table.m_tables[0].keys + pos, key);
                    if(key == cuckoo_hash_table<Key, TableNum>::data_layout_t::empty_key) break;
                    pos = hash_funcs::hash2<Key>(key) % table.m_table_size;
                    key = atomicExch(table.m_tables[1].keys + pos, key);
                    if(key == cuckoo_hash_table<Key, TableNum>::data_layout_t::empty_key) break;
                    pos = hash_funcs::hash3<Key>(key) % table.m_table_size;
                    key = atomicExch(table.m_tables[2].keys + pos, key);
                    if(key == cuckoo_hash_table<Key, TableNum>::data_layout_t::empty_key) break;
                }
            }
            if(iter == table.m_max_evict_num)
                printf("insert failed\n");
        }
    }

    template<std::unsigned_integral Key, std::size_t TableNum>
    __global__ void search_kernel(const cuckoo_hash_table<Key, TableNum> table, const Key* keys, bool* result, const std::size_t num) {
        Key key;
        std::size_t idx;
        std::size_t pos;
        for(idx = blockDim.x * blockIdx.x + threadIdx.x ; idx < num ; idx += gridDim.x * blockDim.x) {
            key = keys[idx];
            if constexpr (TableNum == 2) {
                pos = hash_funcs::hash1<Key>(key) % table.m_table_size;
                if(table.m_tables[0].keys[pos] == key) {
                    result[idx] = true;
                    continue;
                }

                pos = hash_funcs::hash2<Key>(key) % table.m_table_size;
                if(table.m_tables[1].keys[pos] == key) {
                    result[idx] = true;
                    continue;
                }

                result[idx] = false;
            } else if constexpr (TableNum == 3) {
                pos = hash_funcs::hash1<Key>(key) % table.m_table_size;
                if(table.m_tables[0].keys[pos] == key) {
                    result[idx] = true;
                    continue;
                }

                pos = hash_funcs::hash2<Key>(key) % table.m_table_size;
                if(table.m_tables[1].keys[pos] == key) {
                    result[idx] = true;
                    continue;
                }

                pos = hash_funcs::hash3<Key>(key) % table.m_table_size;
                if(table.m_tables[2].keys[pos] == key) {
                    result[idx] = true;
                    continue;
                }

                result[idx] = false;
            }
        }
    }
}
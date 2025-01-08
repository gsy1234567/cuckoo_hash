#pragma once

#include "data_layout.cuh"
#include "utils.hpp"
#include "hash_helper.cuh"
#include <cooperative_groups.h>
#include <chrono>
#include <cmath>

template<std::unsigned_integral Key, std::size_t TableNum, std::size_t GroupSize>
class dycuckoo_hash_table;

namespace {
    template<std::unsigned_integral Key, std::size_t TableNum, std::size_t GroupSize>
    __global__ void insert_kernel(const dycuckoo_hash_table<Key, TableNum, GroupSize> table, const Key* keys, const std::size_t num);

    template<std::unsigned_integral Key, std::size_t TableNum, std::size_t GroupSize>
    __global__ void search_kernel(const dycuckoo_hash_table<Key, TableNum, GroupSize> table, const Key* keys, const std::size_t num, u32* const founded);
}

template<std::unsigned_integral Key, std::size_t TableNum, std::size_t GroupSize>
class dycuckoo_hash_table {
    public:
        using key_type = Key;
        using data_layout_t = DataLayout<key_type>;
        using single_table_t = data_layout_t::dev_single_table;
        using error_table_t = data_layout_t::dev_error_table;
        using host_error_table_t = data_layout_t::host_error_table;
        using hasher_t = hash_funcs::hasher<Key, TableNum>;


        static constexpr std::size_t table_num = TableNum;
        static constexpr std::size_t group_size = GroupSize;
        static constexpr u32 InsertionError = 1_u32;

        std::size_t m_used_size = 0;
        std::size_t m_table_size = 0;
        std::size_t m_max_evict_num = 0;
        std::size_t m_error_table_size = 0;
        single_table_t m_tables[table_num];
        error_table_t m_error_table;
        hasher_t m_hashers; 

    public:
        dycuckoo_hash_table() = default;

        __host__ static void init(dycuckoo_hash_table& table, std::size_t table_size, std::size_t max_evict_num, std::size_t error_table_size) {
            table.m_used_size = 0;
            table.m_table_size = table_size;
            table.m_max_evict_num = max_evict_num;
            table.m_error_table_size = error_table_size;

            for(auto table_idx = 0_sz ; table_idx < table_num ; ++table_idx) {
                single_table_t::init(table.m_tables[table_idx], table_size);
            }

            error_table_t::init(table.m_error_table, error_table_size);

            table.m_hashers.init();

            CUDA_CHECK(cudaStreamSynchronize(0));
        }

        __host__ static void update(dycuckoo_hash_table& ht, std::default_random_engine& egn) {
            ht.m_hashers.update(egn);
        }

        __host__ static void clear(dycuckoo_hash_table& ht) {
            for(auto table_idx = 0_sz ; table_idx < ht.table_num ; ++table_idx) {
                single_table_t::clear(ht.m_tables[table_idx], ht.m_table_size);
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
        }

        __host__ static void check_error(dycuckoo_hash_table& ht, host_error_table_t& host_error_table) {
            host_error_table = host_error_table_t(ht.m_error_table, ht.m_error_table_size);
        }

        __host__ static std::chrono::nanoseconds insert(dycuckoo_hash_table& ht, const key_type* keys, std::size_t num) {
            key_type* dev_keys = nullptr;
            cuda_timer timer;
            std::chrono::nanoseconds ns;

            timer.init();
            init_dev_data_async<key_type>(dev_keys, keys, num, 0);
            CUDA_CHECK(cudaStreamSynchronize(0));

            timer.start(0);
            launch_cg_kernel((const void*)&insert_kernel<Key, TableNum, GroupSize>, 0, 0, ht, dev_keys, num);
            timer.finish(0);

            CUDA_CHECK(cudaStreamSynchronize(0));
            ns = timer.duration();
            deinit_dev_data_async<key_type>(dev_keys);
            CUDA_CHECK(cudaStreamSynchronize(0));
            timer.deinit();

            return ns;
        }

        __host__ static std::chrono::nanoseconds search(dycuckoo_hash_table& ht, const key_type* keys, std::size_t key_len, u32* compressed_found, std::size_t found_len) {
            key_type* dev_keys = nullptr;
            u32* dev_found = nullptr;
            cuda_timer timer;
            std::chrono::nanoseconds ns;

            timer.init();
            init_dev_data_async<key_type>(dev_keys, keys, key_len, 0);
            init_dev_data_async<u32>(dev_found, nullptr, found_len, 0);
            if(found_len * std::numeric_limits<u32>::digits < key_len)
                throw std::invalid_argument("`found_len` * 32 must be greater than or equal to `key_len!");
            CUDA_CHECK(cudaMemsetAsync(dev_found, 0, found_len * sizeof(u32), 0));
            CUDA_CHECK(cudaStreamSynchronize(0));

            timer.start(0);
            launch_cg_kernel((const void*)&search_kernel<Key, TableNum, GroupSize>, 0, 0, ht, dev_keys, key_len, dev_found);
            timer.finish(0);

            CUDA_CHECK(cudaStreamSynchronize(0));
            ns = timer.duration();

            CUDA_CHECK(cudaMemcpyAsync(compressed_found, dev_found, found_len * sizeof(u32), cudaMemcpyDeviceToHost, 0));

            deinit_dev_data_async<key_type>(dev_keys, 0);
            deinit_dev_data_async<u32>(dev_found, 0);
            CUDA_CHECK(cudaStreamSynchronize(0));
            timer.deinit();

            return ns;
        }

        __host__ static void deinit(dycuckoo_hash_table& table) {
            table.m_table_size = 0;
            table.m_max_evict_num = 0;
            table.m_error_table_size = 0;

            for(auto table_idx = 0_sz ; table_idx < table_num ; ++table_idx) {
                single_table_t::deinit(table.m_tables[table_idx]);
            }

            error_table_t::deinit(table.m_error_table);

            CUDA_CHECK(cudaStreamSynchronize(0));
        }
};

namespace {
    template<std::unsigned_integral Key, std::size_t TableNum, std::size_t GroupSize>
    __global__ void insert_kernel(const dycuckoo_hash_table<Key, TableNum, GroupSize> table, const Key* keys, const std::size_t num) {
        namespace cg = cooperative_groups;
        Key key;
        std::size_t pos;
        std::size_t idx;
        std::size_t iter;
        auto tile = cg::tiled_partition<GroupSize>(cg::this_thread_block());

        for(idx = blockDim.x * blockIdx.x + threadIdx.x ; idx < num ; idx += gridDim.x * blockDim.x) {
                
            key = keys[idx];

            for(iter = 0_sz ; iter != table.m_max_evict_num ; ++iter) {
                if constexpr (TableNum >= 1) {
                    pos = table.m_hashers.hash1(key) % table.m_table_size;
                    key = atomicExch(table.m_tables[0].keys + pos, key);
                    if(key == dycuckoo_hash_table<Key, TableNum, GroupSize>::data_layout_t::empty_key) break;
                }

                if constexpr (TableNum >= 2) {
                    pos = table.m_hashers.hash2(key) % table.m_table_size;
                    key = atomicExch(table.m_tables[1].keys + pos, key);
                    if(key == dycuckoo_hash_table<Key, TableNum, GroupSize>::data_layout_t::empty_key) break;
                }

                if constexpr (TableNum >= 3) {
                    pos = table.m_hashers.hash3(key) % table.m_table_size;
                    key = atomicExch(table.m_tables[2].keys + pos, key);
                    if(key == dycuckoo_hash_table<Key, TableNum, GroupSize>::data_layout_t::empty_key) break; 
                }
            }

            u32 failedMask = tile.ballot(iter == table.m_max_evict_num);

            if(failedMask) {
                std::size_t error_ptr;
                if(tile.thread_rank() == 0) {
                    error_ptr = atomicAdd(table.m_error_table.error_ptr, (std::size_t)__popc(failedMask));
                }
                error_ptr = tile.shfl(error_ptr, 0);
                if(failedMask & (1_u32 << tile.thread_rank())) {
                    error_ptr += __popc(failedMask & ((1_u32 << tile.thread_rank()) - 1_u32));
                    if(error_ptr < table.m_error_table_size) {
                        table.m_error_table.keys[error_ptr] = key;
                        table.m_error_table.errors[error_ptr] = dycuckoo_hash_table<Key, TableNum, GroupSize>::InsertionError;
                    }
                }
            }
        }
    }

    template<std::unsigned_integral Key, std::size_t TableNum, std::size_t GroupSize>
    __global__ void search_kernel(const dycuckoo_hash_table<Key, TableNum, GroupSize> table, const Key* keys, const std::size_t num, u32* const founded) {
        namespace cg = cooperative_groups;

        Key key;
        std::size_t idx;
        std::size_t pos;
        auto tile = cg::tiled_partition<GroupSize>(cg::this_thread_block());

        u32 group_found = 0;

        for(idx = blockDim.x * blockIdx.x + threadIdx.x ; idx < num ; idx += gridDim.x * blockDim.x) {
            key = keys[idx];

            if constexpr (TableNum >= 1) {
                pos = table.m_hashers.hash1(key) % table.m_table_size;
                if(table.m_tables[0].keys[pos] == key) {
                    group_found = 1;
                    goto End;
                }
            }

            if constexpr (TableNum >= 2) {
                pos = table.m_hashers.hash2(key) % table.m_table_size;
                if(table.m_tables[1].keys[pos] == key) {
                    group_found = 1;
                    goto End;
                }
            }

            if constexpr (TableNum >= 3) {
                pos = table.m_hashers.hash3(key) % table.m_table_size;
                if(table.m_tables[2].keys[pos] == key) {
                    group_found = 1;
                    goto End;
                }
            }

            End:
            group_found = tile.ballot(group_found);

            if(tile.thread_rank() == 0) {
                atomicOr(founded + idx / std::numeric_limits<u32>::digits, group_found << (GroupSize * (tile.meta_group_rank() % (32 / GroupSize))));
            }  
        }
    }
}
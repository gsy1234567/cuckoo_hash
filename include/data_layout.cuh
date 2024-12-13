#pragma once

#include <concepts>
#include <numeric>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector> 
#include "cuda_helper.cuh"
#include "types.hpp"

template<std::unsigned_integral Key>
class DataLayout {
    public:
        using key_type = Key;
        using error_t = u32;

        static constexpr std::uint8_t key_bits = std::numeric_limits<key_type>::digits;
        static constexpr std::size_t key_size = sizeof(key_type);

        static constexpr key_type empty_key = std::numeric_limits<key_type>::max();
    public:

        struct dev_single_table {
            key_type *keys = nullptr;

            private:
                class dev_single_table_error : public std::runtime_error {
                    public:
                        dev_single_table_error(const std::string& what_arg) : std::runtime_error("dev_single_table_error\n\t" + what_arg) {}
                        const char* what() const noexcept { return std::runtime_error::what(); }
                };

            public:

                static void init(dev_single_table& table, std::size_t single_table_size) {
                    if(table.keys)
                        throw dev_single_table_error("keys already initialized!");
                    CUDA_CHECK(cudaMalloc((void**)&table.keys, single_table_size * key_size));
                    CUDA_CHECK(cudaMemsetAsync((void*)table.keys, 0xFF, single_table_size * key_size));
                }

                static void clear(dev_single_table& table, std::size_t single_table_size) {
                    CUDA_CHECK(cudaMemsetAsync((void*)table.keys, 0xFF, single_table_size * key_size));
                }

                static void deinit(dev_single_table& table) {
                    if(!table.keys)
                        throw dev_single_table_error("keys has been deinitialized!");
                    CUDA_CHECK(cudaFree((void*)table.keys));
                    table.keys = nullptr;
                }
        };

        struct dev_error_table {
            key_type *keys = nullptr;
            error_t *errors = nullptr;
            unsigned long long *error_ptr = nullptr;

            private:

                class dev_error_table_error : public std::runtime_error {
                    public:
                        dev_error_table_error(const std::string& what_arg) : std::runtime_error("dev_error_table_error\n\t" + what_arg) {}
                        const char* what() const noexcept { return std::runtime_error::what(); }
                };
            
            public:
                static void init(dev_error_table& table, std::size_t error_table_size) {
                    if(table.keys || table.errors || table.error_ptr)
                        throw dev_error_table_error("keys or errors or error_ptr already initialized!");
                    CUDA_CHECK(cudaMalloc((void**)&table.keys, error_table_size * key_size));
                    CUDA_CHECK(cudaMalloc((void**)&table.errors, error_table_size * sizeof(error_t)));
                    CUDA_CHECK(cudaMalloc((void**)&table.error_ptr, sizeof(unsigned long long)));
                    CUDA_CHECK(cudaMemsetAsync((void*)table.error_ptr, 0, sizeof(unsigned long long)));
                }

                static void deinit(dev_error_table& table) {
                    if(!table.keys || !table.errors || !table.error_ptr)
                        throw dev_error_table_error("keys or errors or error_ptr has been deinitialized!");
                    CUDA_CHECK(cudaFree((void*)table.keys));
                    CUDA_CHECK(cudaFree((void*)table.errors));
                    CUDA_CHECK(cudaFree((void*)table.error_ptr));
                    table.keys = nullptr;
                    table.errors = nullptr;
                    table.error_ptr = nullptr;
                }
        };

        struct host_error_table {
            std::vector<key_type> keys;
            std::vector<error_t> errors;
            unsigned long long size;

            private:

                class host_error_table_error : public std::runtime_error {
                    public:
                        host_error_table_error(const std::string& what_arg) : std::runtime_error("host_error_table_error\n\t" + what_arg) {}
                        const char* what() const noexcept { return std::runtime_error::what(); }
                };

            public:
                host_error_table() = default;
                host_error_table(const dev_error_table& dev_table, unsigned long long max_len) {
                    if(!dev_table.keys || !dev_table.errors || !dev_table.error_ptr)
                        throw std::invalid_argument("keys or errors or error_ptr has not been initialized!");
                    CUDA_CHECK(cudaMemcpy(&size, dev_table.error_ptr, sizeof(size), cudaMemcpyDeviceToHost));
                    size = std::min(size, max_len);
                    keys.resize(size);
                    errors.resize(size);
                    CUDA_CHECK(cudaMemcpy(keys.data(), dev_table.keys, size * key_size, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(errors.data(), dev_table.errors, size * sizeof(error_t), cudaMemcpyDeviceToHost));
                } 

                inline bool empty() const { return size == 0; }
        };
};
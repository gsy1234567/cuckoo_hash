#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdexcept>
#include <chrono>

class cuda_runtime_error : public std::runtime_error {
    public:
        cuda_runtime_error(const std::string& what_arg) : std::runtime_error("cuda runtime error:\n\t" + what_arg) {}
        const char* what() const noexcept { return std::runtime_error::what(); }
};

inline void cuda_check(cudaError_t err, int line, const char* file) {
    char errBuf[512];
    if(err != cudaSuccess) {
        snprintf(errBuf, sizeof(errBuf), "%s:%d\n\tdecs %s", file, line, cudaGetErrorString(err));
        throw cuda_runtime_error(errBuf);
    }
}

#define CUDA_CHECK(err) cuda_check(err, __LINE__, __FILE__)

template<typename T>
void init_dev_data(T* &p_dev, const T* p_host, std::size_t len) {
    if(p_dev)
        throw std::runtime_error("p_dev must be a nullptr!");

    CUDA_CHECK(cudaMalloc((void**)&p_dev, len * sizeof(T)));
    CUDA_CHECK(cudaMemcpy((void*)p_dev, (const void*)p_host, len * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void init_dev_data_async(T* &p_dev, const T* p_host, std::size_t len, cudaStream_t stream = 0) {
    if(p_dev)
        throw std::runtime_error("p_dev must be a nullptr!");

    CUDA_CHECK(cudaMallocAsync((void**)&p_dev, len * sizeof(T), stream));

    if(p_host)
        CUDA_CHECK(cudaMemcpyAsync((void*)p_dev, (const void*)p_host, len * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template<typename T>
void deinit_dev_data(T* &p_dev) {
    if(!p_dev)
        throw std::runtime_error("p_dev must be a valid pointer!");
    CUDA_CHECK(cudaFree((void*)p_dev));
    p_dev = nullptr;
}

template<typename T>
void deinit_dev_data_async(T* &p_dev, cudaStream_t stream = 0) {
    if(!p_dev)
        throw std::runtime_error("p_dev must be a valid pointer!");
    CUDA_CHECK(cudaFreeAsync((void*)p_dev, stream));
    p_dev = nullptr;
}

template<typename... Args>
void launch_cg_kernel(const void* kernel, std::size_t dySharedMem, cudaStream_t stream, Args&&... args) {
    int grid_size, block_size;
    void *params[sizeof...(args)] = {reinterpret_cast<void*>(&args)...};
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, dySharedMem));
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        kernel, 
        dim3{static_cast<unsigned>(grid_size), 1, 1}, 
        dim3{static_cast<unsigned>(block_size), 1, 1}, 
        (void**)&params,
        dySharedMem, 
        stream
    ));
}

class cuda_timer {
    private:
        cudaEvent_t m_start, m_stop;
    public:
        cuda_timer() = default;

        inline void init() {
            CUDA_CHECK(cudaEventCreate(&m_start));
            CUDA_CHECK(cudaEventCreate(&m_stop));
        }

        inline void start(cudaStream_t stream = 0) {
            CUDA_CHECK(cudaEventRecord(m_start, stream));
        }

        inline void finish(cudaStream_t stream = 0) {
            CUDA_CHECK(cudaEventRecord(m_stop, stream));
        }

        inline std::chrono::nanoseconds duration() {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, m_start, m_stop));
            return std::chrono::nanoseconds(static_cast<int64_t>(ms * 1e6));
        } 

        inline void deinit() {
            CUDA_CHECK(cudaEventDestroy(m_start));
            CUDA_CHECK(cudaEventDestroy(m_stop));
        }
};
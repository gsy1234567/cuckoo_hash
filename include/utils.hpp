#pragma once

#include "types.hpp"
#include <concepts>
#include <cstddef>

template<typename T>
concept location = std::is_same_v<T, host> || std::is_same_v<T, device>;

__host__ __device__ constexpr std::size_t operator""_KB(unsigned long long int n) { return n * 1024; }
__host__ __device__ constexpr std::size_t operator""_MB(unsigned long long int n) { return n * 1024_KB; }
__host__ __device__ constexpr std::size_t operator""_GB(unsigned long long int n) { return n * 1024_MB; }

__host__ __device__ constexpr std::size_t operator""_K(unsigned long long int n) { return n * 1024; }
__host__ __device__ constexpr std::size_t operator""_M(unsigned long long int n) { return n * 1024_K; }
__host__ __device__ constexpr std::size_t operator""_G(unsigned long long int n) { return n * 1024_M; }

__host__ __device__ constexpr u32 operator""_u32(unsigned long long int n) { return static_cast<u32>(n); }
__host__ __device__ constexpr u64 operator""_u64(unsigned long long int n) { return static_cast<u64>(n); }
__host__ __device__ constexpr std::size_t operator""_sz(unsigned long long int n) { return static_cast<std::size_t>(n); }
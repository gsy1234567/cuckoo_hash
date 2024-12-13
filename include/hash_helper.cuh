#pragma once

#include <concepts>
#include <random>
#include "utils.hpp"

namespace hash_funcs {

    template<std::unsigned_integral T>
    __device__ inline T hash1(T key);
    
    template<>
    __device__ inline u32 hash1<u32>(u32 key) {
        key = ~key + (key << 15_u32);
        key = key ^ (key >> 12_u32);
        key = key + (key << 2_u32);
        key = key ^ (key >> 4_u32);
        key = key * 2057_u32;
        key = key ^ (key >> 16_u32);
        return key;
    }

    template<std::unsigned_integral T>
    __device__ inline T hash2(T key);

    template<>
    __device__ inline u32 hash2<u32>(u32 key) {
        key = (key + 0x7ed55d16_u32) + (key << 12_u32);
        key = (key ^ 0xc761c23c_u32) ^ (key >> 19_u32);
        key = (key + 0x165667b1_u32) + (key << 5_u32);
        key = (key + 0xd3a2646c_u32) ^ (key << 9_u32);
        key = (key + 0xfd7046c5_u32) + (key << 3_u32);
        key = (key ^ 0xb55a4f09_u32) ^ (key >> 16_u32);
        return key;
    }

    template<std::unsigned_integral T>
    __device__ inline T hash3(T key);

    template<>
    __device__ inline u32 hash3<u32>(u32 key) {
        return ((key ^ 59064253_u32) + 72355969_u32) % 294967291_u32;
    }

    template<std::unsigned_integral T>
    __device__ inline T hash4(T key);

    template<>
    __device__ inline u32 hash4(u32 key) {
        key = (key ^ 61_u32) ^ (key >> 16_u32);
        key = key + (key << 3_u32);
        key = key ^ (key >> 4_u32);
        key = key * 0x27d4eb2d_u32;
        key = key ^ (key >> 15_u32);
        return key;
    }

    template<std::unsigned_integral T>
    __device__ inline T hash5(T key);

    template<>
    __device__ inline u32 hash5<u32>(u32 key) {
        key -= (key << 6_u32);
        key ^= (key >> 17_u32);
        key -= (key << 9_u32);
        key ^= (key << 4_u32);
        key -= (key << 3_u32);
        key ^= (key << 10_u32);
        key ^= (key >> 15_u32);
        return key;
    }

    template<std::unsigned_integral T>
    __host__ inline T rand_shift(std::default_random_engine& egn) {
        std::uniform_int_distribution<T> dist(1, std::numeric_limits<T>::digits - 1);
        return dist(egn);
    }

    template<std::unsigned_integral T>
    __host__ inline T rand_num(std::default_random_engine& egn) {
        std::uniform_int_distribution<T> dist(1, std::numeric_limits<T>::max());
        return dist(egn);
    }

    template<std::unsigned_integral T>
    struct dy_hash_1;

    template<>
    struct dy_hash_1<u32> {
        u32 shift1 = 15, shift2 = 12, shift3 = 2, shift4 = 4, shift5 = 16;
        u32 num = 2057;

        __host__ void update(std::default_random_engine& egn) {
            shift1 = rand_shift<u32>(egn);
            shift2 = rand_shift<u32>(egn);
            shift3 = rand_shift<u32>(egn);
            shift4 = rand_shift<u32>(egn);
            shift5 = rand_shift<u32>(egn);
            num = rand_num<u32>(egn);
        }

        __host__ void init() {
            shift1 = 15;
            shift2 = 12;
            shift3 = 2;
            shift4 = 4;
            shift5 = 16;
            num = 2057;
        }

        __host__ __device__ u32 operator()(u32 key) const {
            key = ~key + (key << shift1);
            key = key ^ (key >> shift2);
            key = key + (key << shift3);
            key = key ^ (key >> shift4);
            key = key * num;
            key = key ^ (key >> shift5);
            return key;
        }
    };

    template<std::unsigned_integral T>
    struct dy_hash_2;

    template<>
    struct dy_hash_2<u32> {
        u32 num1 = 0x7ed55d16, num2 = 0xc761c23c, num3 = 0x165667b1, num4 = 0xd3a2646c, num5 = 0xfd7046c5, num6 = 0xb55a4f09;
        u32 shift1 = 12, shift2 = 19, shift3 = 5, shift4 = 9, shift5 = 3, shift6 = 16;

        __host__ void init() {
            num1 = 0x7ed55d16;
            num2 = 0xc761c23c;
            num3 = 0x165667b1;
            num4 = 0xd3a2646c;
            num5 = 0xfd7046c5;
            num6 = 0xb55a4f09;
            shift1 = 12;
            shift2 = 19;
            shift3 = 5;
            shift4 = 9;
            shift5 = 3;
            shift6 = 16;
        }

        __host__ void update(std::default_random_engine& egn) {
            num1 = rand_num<u32>(egn);
            num2 = rand_num<u32>(egn);
            num3 = rand_num<u32>(egn);
            num4 = rand_num<u32>(egn);
            num5 = rand_num<u32>(egn);
            num6 = rand_num<u32>(egn);

            shift1 = rand_shift<u32>(egn);
            shift2 = rand_shift<u32>(egn);
            shift3 = rand_shift<u32>(egn);
            shift4 = rand_shift<u32>(egn);
            shift5 = rand_shift<u32>(egn);
            shift6 = rand_shift<u32>(egn);
        }

        __host__ __device__ u32 operator()(u32 key) const {
            key = (key + num1) + (key << shift1);
            key = (key ^ num2) ^ (key >> shift2);
            key = (key + num3) + (key << shift3);
            key = (key + num4) + (key << shift4);
            key = (key + num5) + (key << shift5);
            key = (key ^ num6) ^ (key >> shift6);
            return key;
        }
    };

    template<std::unsigned_integral T>
    struct dy_hash_5;

    template<>
    struct dy_hash_5<u32> {
        u32 shift1 = 6, shift2 = 17, shift3 = 9, shift4 = 4, shift5 = 3, shift6 = 10, shift7 = 15;
        
        __host__ void init() {
            shift1 = 6;
            shift2 = 17;
            shift3 = 9;
            shift4 = 4;
            shift5 = 3;
            shift6 = 10;
            shift7 = 15;
        }

        __host__ void update(std::default_random_engine& egn) {
            shift1 = rand_shift<u32>(egn);
            shift2 = rand_shift<u32>(egn);
            shift3 = rand_shift<u32>(egn);
            shift4 = rand_shift<u32>(egn);
            shift5 = rand_shift<u32>(egn);
            shift6 = rand_shift<u32>(egn);
            shift7 = rand_shift<u32>(egn);
        }

        __host__ __device__ u32 operator()(u32 key) const {
            key -= (key << shift1);
            key ^= (key >> shift2);
            key -= (key << shift3);
            key ^= (key << shift4);
            key -= (key << shift5);
            key ^= (key >> shift6);
            key -= (key << shift7);
            return key;
        }
    };

    template<std::unsigned_integral T, std::size_t N>
    struct hasher;

    template<std::unsigned_integral T>
    struct hasher<T, 2> : public dy_hash_1<T>, dy_hash_2<T> {

        __host__ void init() {
            dy_hash_1<T>::init();
            dy_hash_2<T>::init();
        }

        __host__ void update(std::default_random_engine& egn) {
            dy_hash_1<T>::update(egn);
            dy_hash_2<T>::update(egn);
        }

        __host__ __device__ T hash1(T key) const { return dy_hash_1<T>::operator()(key); }
        __host__ __device__ T hash2(T key) const { return dy_hash_2<T>::operator()(key); }
    };

    template<std::unsigned_integral T>
    struct hasher<T, 3> : public dy_hash_1<T>, dy_hash_2<T>, dy_hash_5<T> {
        __host__ void init() {
            dy_hash_1<T>::init();
            dy_hash_2<T>::init();
            dy_hash_5<T>::init();
        }

        __host__ void update(std::default_random_engine& egn) {
            dy_hash_1<T>::update(egn);
            dy_hash_2<T>::update(egn);
            dy_hash_5<T>::update(egn);
        }

        __host__ __device__ T hash1(T key) const { return dy_hash_1<T>::operator()(key); }
        __host__ __device__ T hash2(T key) const { return dy_hash_2<T>::operator()(key); }
        __host__ __device__ T hash3(T key) const { return dy_hash_5<T>::operator()(key); }
    };

}

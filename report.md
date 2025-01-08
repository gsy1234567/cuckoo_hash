# Cuckoo Hash Table Report

## Task1
Create a hash table of size $2^{25}$ in GPU global memory, where each table entry stores a 32-bit integer. Insert a set of $2^s$ random integer keys into hash table, for $s=11, 12, 13, \cdots, 23, 24$.

| # of keys | insert2 speeds (M times/sec) | insert3 speeds (M times/sec) |
|-----------|------------------------------|------------------------------|
| $2^{11}$ | 140.432681 | 135.525924 |
| $2^{12}$ | 246.164941 | 249.889423 |
| $2^{13}$ | 438.362029 | 432.443846 |
| $2^{14}$ | 695.895938 | 697.940880 |
| $2^{15}$ | 1080.480426| 1069.460423|
| $2^{16}$ | 1370.845274| 1392.975185|
| $2^{17}$ | 1544.512921| 1586.069616|
| $2^{18}$ | 1662.255970| 1661.334199|
| $2^{19}$ | 1702.899998| 1719.745296|
| $2^{20}$ | 1717.627585| 1738.125637|
| $2^{21}$ | 1720.139533| 1723.532048|
| $2^{22}$ | 1677.893752| 1680.135787|
| $2^{23}$ | 1588.809516| 1591.950114|
| $2^{24}$ | 1453.404369| 1477.580447|

## Task2 
Insert a set $\mathbb{S}$ of $2^{24}$ random keys into a hash table of size $2^{25}$, then perform lookups for the following sets of keys $\mathbb{S}_0, \cdots, \mathbb{S}_{10}$. Each set $\mathbb{S}_{i}$ should contain $2^{24}$ keys, where $(100-10i)$ percent of the keys are randomly chosen from $\mathbb{S}$, and the remainder are random 32-bit keys.

| i |  insert2 speeds (M times/sec) | insert3 speeds (M times/sec) | search2 speeds (M times/sec) | search3 speeds (M times/sec) |
|---|-------------------------------|------------------------------|------------------------------|------------------------------|
| 0 | 1297.083598 | 1467.004711 | 4001.725225 | 3893.449831 |
| 1 | 1466.831239 | 1491.691122 | 3758.397958 | 3414.356557 |
| 2 | 1470.479434 | 1494.745605 | 3561.382644 | 3079.156916 |
| 3 | 1470.025871 | 1494.524172 | 3391.940822 | 2809.015640 | 
| 4 | 1469.707504 | 1494.436775 | 3256.983608 | 2586.088274 | 
| 5 | 1470.587608 | 1494.448456 | 3134.161670 | 2399.858187 | 
| 6 | 1469.715744 | 1494.014114 | 3027.180991 | 2248.392436 | 
| 7 | 1469.716742 | 1494.007694 | 2930.227615 | 2114.772416 | 
| 8 | 1470.611358 | 1494.225914 | 2857.163342 | 2001.005797 |
| 9 | 1469.889802 | 1494.411217 | 2800.579272 | 1906.087786 |
| 10| 1470.920800 | 1493.232235 | 2770.756501 | 1846.183845 |

## Details for Task1 and Task2
- I found that there is no insertion failed in `Task1` and `Task2`, so I use static hash function.
- Here I show my hash functions, you can also find them in file `include/hash_helper.cuh`

```c
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
```
- Using static hash function avoid load hash function parameters from global memory, then we can get better performance, however, if any insertion operation failed, we need to use another group function, which static hash is not support.

## Task3
Fix a set of $n=2^{24}$ random keys, and measure the time to insert the keys into hash tables of sizes $1.1n, 1.2n, \cdots, 2n$. Also, measure the insertion times for hash tables of sizes $1.01n, 1.02n, 1.05n$. Terminate the experiment if it takes too long. Also if I change hash function more than 30 times, I will terminate the experiment.

| table size ($\times n$) | insert2 speeds (M times/sec) | search2 speeds (M times/sec) | insert3 speeds (M times/sec) | search3 speeds (M times/sec) |
|-------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| 1.01 | failed | failed | 658.086643 | 4115.239347 | 
| 1.02 | failed | failed | 659.994325 | 4121.554139 |
| 1.05 | 571.972941 | 4362.320293 | 664.615803 | 4146.693633 |
| 1.1  | 596.176387 | 4384.852478 | 671.719073 | 4173.098400 | 
| 1.2  | 631.605057 | 4430.383989 | 684.515242 | 4222.789885 |
| 1.3  | 655.168558 | 4478.094029 | 695.928482 | 4272.229792 |
| 1.4  | 673.817580 | 4525.467165 | 705.464966 | 4298.683449 |
| 1.5  | 688.590535 | 4550.858779 | 714.471468 | 4330.268141 | 
| 1.6  | 701.034349 | 4582.553014 | 722.184480 | 4366.389297 |
| 1.7  | 711.934214 | 4610.665668 | 729.372771 | 4407.856493 |
| 1.8  | 721.274430 | 4641.360415 | 736.272726 | 4440.470238 |
| 1.9  | 729.090590 | 4665.813109 | 741.584197 | 4459.445340 |
| 2.0  | 736.344283 | 4674.065248 | 747.252481 | 4476.136672 |

## Task4
Using $n=2^{24}$ random keys and a hash table of size $1.4n$, experiment with different bounds on the maximum length of an eviction chain before restarting. Which bound gives the best running time for constructing the hash table?

| evict chain length | insert2 speeds (M times/sec) | insert3 speeds (M times/sec) |
|--------------------|------------------------------|------------------------------|
| 10 | failed     |  704.723627 | 
| 20 | 673.340236 |  705.080172 |
| 30 | 673.584865 |  705.310473 | 
| 40 | 673.603055 |  705.299479 |
| 50 | 673.602887 |  705.280284 | 
| 60 | 673.469965 |  705.233442 |
| 70 | 673.286787 |  705.197005 |
| 80 | 673.637144 |  705.330773 |
| 90 | 673.659308 |  704.990488 | 
| 100| 673.726652 |  705.158882 |

- In my point of view, maximum length of eviction chain only affects whether or not we can construct the hash table, but if we can construct the hash table, it doesn't affect the insertion speed.

## Details for Task3 and Task4
- Here I show my hash functions, you can also find them in file `include/hash_helper.cuh`
```c
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
```

- I also optimize the data format to store the find result. Instead of using a `bool` to indicate whether we find the key or not, I use only a bit to indicate whether we find the key or not. Since 32 threads in a warp is excuted in parallel, we can use a `u32` to store their finding results. In this way, we can reduce global memory write times, then we can get better performance. Here I show my code, you can also find them in file `include/dycuckoo.cuh`
```c
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
```

## Device Info
Tesla V100-PCIE-32GB
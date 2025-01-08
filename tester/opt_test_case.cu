#include "opt_test_case.cuh"
#include <iostream>


void OptTestCase::run(std::span<const u32> insert_keys, std::span<const u32> search_keys , std::span<const std::size_t> max_evicts) {
    printf("Run OptTestCase\n");
    // run_single<2, 2>(insert_keys, search_keys, max_evicts);
    // run_single<2, 4>(insert_keys, search_keys, max_evicts);
    // run_single<2, 8>(insert_keys, search_keys, max_evicts);
    // run_single<2, 16>(insert_keys, search_keys, max_evicts);
    run_single<2, 32>(insert_keys, search_keys, max_evicts);
    // run_single<3, 2>(insert_keys, search_keys, max_evicts);
    // run_single<3, 4>(insert_keys, search_keys, max_evicts);
    // run_single<3, 8>(insert_keys, search_keys, max_evicts);
    // run_single<3, 16>(insert_keys, search_keys, max_evicts);
    run_single<3, 32>(insert_keys, search_keys, max_evicts);
}
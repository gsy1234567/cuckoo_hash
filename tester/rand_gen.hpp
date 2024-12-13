#pragma once

#include <unordered_set>
#include <vector>
#include <cstdint>
#include <random>
#include <numeric>
#include <ctime>
#include <fstream>
#include <string>
#include <span>

void rand_gen(std::vector<std::uint32_t>& data, std::size_t size) {
    std::unordered_set<std::uint32_t> set;
    std::default_random_engine egn;
    egn.seed(std::time(nullptr));
    std::uniform_int_distribution<std::uint32_t> dist(std::numeric_limits<std::uint32_t>::min(), std::numeric_limits<std::uint32_t>::max());
    while(set.size() < size) {
        set.insert(dist(egn));
    }
    data.clear();
    data.assign(set.begin(), set.end());
}

void store(std::span<const std::uint32_t> data, const std::string& fileName) {
    std::ofstream os (fileName, std::ios::binary);
    if(!os) {
        throw std::runtime_error("Could not open the file :" + fileName);
    }
    std::size_t size = data.size();
    os.write((const char*)&size, sizeof(size));
    os.write((const char*)data.data(), sizeof(std::uint32_t) * data.size());
}

void load(std::vector<std::uint32_t>& data, const std::string& fileName) {
    std::ifstream is (fileName, std::ios::binary);
    if(!is) {
        throw std::runtime_error("Could not open the file :" + fileName);
    }
    std::size_t size;
    is.read((char*)&size, sizeof(size));
    data.resize(size);
    is.read(const_cast<char*>((const char*)data.data()), sizeof(std::uint32_t) * size);
}
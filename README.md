# How to generate a random test case?

```c++
#include "tester/rand_gen.hpp" //include the header file.
#include <iostream>


int main(int argc, char** argv) {
    try {
        std::vector<u32> keys;
        rand_gen(keys, 1_M);
        store(keys, "../data/file_name");
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
} 
```
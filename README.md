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

# Prerequest
1. CMake Version >= 3.30
2. cuda toolkit version >= 12

# How to build ?
mkdir build && cd build && cmake ..

- There are four exe in the build folder:
    - task1
    - task2
    - task3
    - task4
You can run them.



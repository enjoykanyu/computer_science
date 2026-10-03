指针和C语言的指针差不多 * &

指针本质是内存地址

```cpp
#include <iostream>


int main()
{
    int var = 8;
    void* ptr = &var;
    std::cout<< ptr <<std::endl;
    std::cin.get();
}
```
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

编译执行
![打印指针地址.png](static/打印指针地址.png)
可以看到打印出来一个16进制的数字，这个是它的地址

给程序打上断点，可以看到对应的指针地址和对应的value
![debug地址和value.png](static/debug地址和value.png)
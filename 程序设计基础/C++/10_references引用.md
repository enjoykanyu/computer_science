引用它和指针很像，都会指向某个数值，但引用本身不创建内存对象

```cpp
#include <iostream>


int main()
{
    int var = 8;
    int& ref = var;
    ref = 9;
    std::cout<< ref <<std::endl;
    std::cin.get();
}
```
可以看到打印了9
![reference打印.png](static/reference打印.png)
ref引用将对应的值改变了，但记住这不是指针
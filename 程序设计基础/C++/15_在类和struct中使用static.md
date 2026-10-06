### 类或者结构体内部使用static
修饰内部变量和方法的时候，被类的所有实例共享同一个静态变量

```cpp
#include <iostream>

struct Entity{

    int x,y;
    void Print(){
        std::cout << x<< ","<<y << std::endl;
    }
};

int main()
{
    Entity e;
    e.x = 3;
    e.y = 10;
    Entity e1 ={10,3};
    e.Print();
    e1.Print();
    std::cin.get();
}
```
在这里定义一个struct，创建它的两个实例
打印
![img_15.png](img_15.png)
这里符合预期
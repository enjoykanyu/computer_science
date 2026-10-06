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

- static修饰结构体内部变量
```cpp
#include <iostream>

struct Entity{

    static int x,y;
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
可以看到编译报错了
![img_16.png](img_16.png)

重新修改程序
```cpp
#include <iostream>

struct Entity{

    static int x,y;
    void Print(){
        std::cout << x<< ","<<y << std::endl;
    }
};


int main()
{
    Entity e;
    e.x = 3;
    e.y = 10;
    Entity e1;
    e1.x = 10; //赋值
    e1.y = 3;
    e.Print();
    e1.Print();
    std::cin.get();
}
```
![img_17.png](img_17.png)
可以看到报错外部链接解析错误找不到x和y
这里可以和之前的链接阶段得定义函数声明来看，x和y被static修饰了，因此外部函数看不到它，因此得单独声明下x和y

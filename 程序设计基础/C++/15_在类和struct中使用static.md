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

```cpp
#include <iostream>

struct Entity{

    static int x,y;
    void Print(){
        std::cout << x<< ","<<y << std::endl;
    }
};

int Entity::x;
int Entity::y;
int main()
{
    Entity e;
    e.x = 3;
    e.y = 10;
    Entity e1;
    e1.x = 10;
    e1.y = 3;
    e.Print();
    e1.Print();
    std::cin.get();
}
```

int Entity::x;
int Entity::y;
单独声明了之后发现build打印了两个10，3
![img_18.png](img_18.png)
这个是因为static修饰的静态变量全局只会共享一个，可以理解为定义了多个entity，相当于指向的都是同一个数据，多个定义的指针地址都是相同的

- static修饰函数访问外部实例变量
```cpp
#include <iostream>

struct Entity{

    int x,y;
    static void Print(){
        std::cout << x<< ","<<y << std::endl;
    }
};

int main()
{
    Entity e;
    e.x = 3;
    e.y = 10;
    Entity e1;
    e1.x = 10;
    e1.y = 3;
    Entity::Print();
    Entity::Print();
    std::cin.get();
}
```
可以看到编译报错了
![img_19.png](img_19.png)
这个是因为静态方法没有类实例，类中定义的每个非静态方法都会将当前类实例作为参数传入，静态方法和外部定义的方法没有本质区别
实际编译出来和如下外部方法相同
```cpp
#include <iostream>

struct Entity{

    int x,y;
    static void Print(){
        std::cout << x<< ","<<y << std::endl;
    }
};
//这里
static void Print(){
        std::cout << x<< ","<<y << std::endl;
    }
int main()
{
    Entity e;
    e.x = 3;
    e.y = 10;
    Entity e1;
    e1.x = 10;
    e1.y = 3;
    Entity::Print();
    Entity::Print();
    std::cin.get();
}
```
当加上Entity参数
```cpp
#include <iostream>

struct Entity{

    int x,y;
    static void Print(){
        std::cout << x<< ","<<y << std::endl;
    }
};
//加上Entity参数这里就不会报错了，可以和之前结合起来理解，类中的所有非静态方法和变量都会作为参数传入
//这里去掉了Entity e相当于给Print函数加上static修饰符，它不知道该给哪个实例的x和y进行赋值
static void Print(Entity e){
        std::cout << e.x<< ","<<e.y << std::endl;
}
int main()
{
    Entity e;
    e.x = 3;
    e.y = 10;
    Entity e1;
    e1.x = 10;
    e1.y = 3;
    Entity::Print();
    Entity::Print();
    std::cin.get();
}
```
加上Entity参数这里就不会报错了，可以和之前结合起来理解，类中的所有非静态方法和变量都会作为参数传入
这里去掉了Entity e相当于给Print函数加上static修饰符，它不知道该给哪个实例的x和y进行赋值

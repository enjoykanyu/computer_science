- 撰写一个打印函数
```cpp
#include <iostream>

class Entity{
 
public:
    float X,Y;
    void Print(){
        std::cout<< X << "," << Y << std::endl;
    }
};

int main()
{
    Entity e;
    e.Print();
    std::cin.get();
}
```
![内存残留打印.png](static/内存残留打印.png)
可以看到打印了两个不规则数据，这是因为当我们实例化该实体并为其分配内存时，并没有真正对那块内存进行初始化，这意味着我们读取到的只是那块内存空间中残留的任意数据。而我们真正想做的，通常是对这块内存进行初始化，将其清零或设置为类似的值，


### 增加Init函数初始化
```cpp
#include <iostream>

class Entity{
 
public:
    float X,Y;
    void Init(){
        X = 0.0f;
        Y = 0.0f;
    }
    void Print(){
        std::cout<< X << "," << Y << std::endl;
    }
};

int main()
{
    Entity e;
    e.Init();
    e.Print();
    std::cin.get();
}
```
在调用类实例函数之前先调用初始化函数赋值
![打印Init函数初始化赋值.png](static/打印Init函数初始化赋值.png)
可以看到打印的数值均为0，符合预期
但这样很麻烦，这里有个构造函数，可以理解和Java差不多，在类被创建的时候会去调用构造函数，可以在构造函数内部将X、Y进行初始化赋值
注意：像Java会给值赋默认值初始化，但C++不会，得手动赋值
```cpp
#include <iostream>

class Entity{
 
public:
    float X,Y;
    Entity(){
        X = 0.0f;
        Y = 0.0f;
    }

    void Print(){
        std::cout<< X << "," << Y << std::endl;
    }
};

int main()
{
    Entity e;
    e.Print();
    std::cin.get();
}
```
![无参构造函数.png](static/无参构造函数.png)
可以看到构造函数赋值0成功了，X、Y都被赋予了0.0
当然构造函数还可以构造很多个构成重载
```cpp
#include <iostream>

class Entity{
 
public:
    float X,Y;
    Entity(){
        X = 0.0f;
        Y = 0.0f;
    }
    Entity(float x,float y){
        X=x;
        Y=y;
    }

    void Print(){
        std::cout<< X << "," << Y << std::endl;
    }
};

int main()
{
    Entity e(3,3);
    e.Print();
    std::cin.get();
}
```
有参构造和无参构造
![有参构造函数调用打印.png](static/有参构造函数调用打印.png)
可以看到这里调用了有参构造X和Y被赋予了3和3



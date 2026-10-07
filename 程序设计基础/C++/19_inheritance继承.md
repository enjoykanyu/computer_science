继承可以减少代码冗余，共同程序定义在父类中，各个子类可以继承同时重写函数和增加专属于自己的函数

```cpp
#include <iostream>

class Entity{
 
public:
    float X,Y;
    Entity(){
        X = 0.0f;
        Y = 0.0f;
        std::cout<< "Created Entity" << std::endl;
    }
    ~Entity(){
        std::cout<< "Destroyed Entity" << std::endl;
    }

    void Print(){
        std::cout<< X << "," << Y << std::endl;
    }
};

class SubEntity: public Entity
{

};
void Function()
{
    Entity e;
    e.Print();
}
int main()
{   
    Function();
    std::cin.get();
}
```
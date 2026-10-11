```cpp
#include<iostream>
class Entity
{
private:
    std::string m_Name;
public:
    const std::string& GetName() const
    {
        return m_Name;
    }
};

int mian(){

    const Entity e;
    e.GetName();
    std::cin.get();
}
```
此时调用e.GetName();不会报错
当剔除GetName的const关键词修饰符
```cpp
#include<iostream>
class Entity
{
private:
    std::string m_Name;
public:
    const std::string& GetName()
    {
        return m_Name;
    }
};

int mian(){

    const Entity e;
    e.GetName();
    std::cin.get();
}
```
会发现报错了
![常量类调用非常量函数报错.png](常量类调用非常量函数报错.png)
这里是因为限制了const类对象与const函数的调用关系，指的是常量函数和类 不可变 const类不可调用非const函数

### mutable
```cpp
#include<iostream>
class Entity
{
private:
    std::string m_Name;
    int debugCount = 0;
public:
    const std::string& GetName() const
    {
        debugCount++;
        return m_Name;
    }
};

int mian(){

    const Entity e;
    e.GetName();
    std::cin.get();
}
```
const函数指的是常量函数，但有的时候会出于调试目的想看某个const常量函数调用的次数，因此会在函数中定义类似于debugCount进行计数操作
但会看到编译器报错了
![常量函数变量修改报错.png](常量函数变量修改报错.png)
```cpp
#include<iostream>
class Entity
{
private:
    std::string m_Name;
    int debugCount = 0;
public:
    const std::string& GetName()
    {
        debugCount++;
        return m_Name;
    }
};

int mian(){

    const Entity e;
    e.GetName();
    std::cin.get();
}
```

将const从函数中移除又可以看到编译器不报错了，但是类对象调用又报错了
![移除const类对象调用报错.png](移除const类对象调用报错.png)
很矛盾，这时候mutable关键字可以解决这个问题
```cpp
#include<iostream>
class Entity
{
private:
    std::string m_Name;
    mutable int debugCount = 0;
public:
    const std::string& GetName() const
    {
        debugCount++;
        return m_Name;
    }
};

int mian(){

    const Entity e;
    e.GetName();
    std::cin.get();
}
```
可以看到编译器完全不报错了

mutable关键字修饰符相当于类内部的 const 方法实际上能够修改这个成员
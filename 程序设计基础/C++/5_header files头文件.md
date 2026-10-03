当有一个函数调用了另一个函数，C++编译器不知道另一个函数在哪里，是否存在，因此需声明下，因此需一个专门存放声明的地方，头文件可以存放各个函数的声明，在调用的时候让编译器知道函数的存在，注意这里只用声明，不是定义，同样的函数签名的函数只可以定义一次

# pragma once 含义

```cpp
# pragma once
void Log(const char* Message);
```

可以看到在之前的头文件定义中有一个pragma once
这里的含义指的是：确保头文件只被引入一次编译单元

### 举个例子
Log.h
```cpp
// # pragma once
void Log(const char* Message);
struct testValue
{
    /* data */
};
```
这里将pragma once注释掉，在math引入头文件两次

```cpp
#include <iostream>
#include "Log.h"
#include "Log.h"

static int Multiply(int a, int b)
{
    Log("Multiply");
    return a * b;
}

int main()
{
    std::cout<< Multiply(8,5) <<std::endl;
    std::cin.get();
}
```
build程序
![pragma%20once为定义导致多次引入报错.png](pragma%20once为定义导致多次引入报错.png)

当取消注释pragma once
```cpp
# pragma once
void Log(const char* Message);
struct testValue
{
    /* data */
};
```

可以看到build不报错了

### ifndef
```cpp
#ifndef _LOG_H
#define _LOG_H
void Log(const char* Message);
struct testValue
{
    /* data */
};
#endif
```
这里的ifndef含义是假设存在_LOG_H未定义则程序将继续执行，如下内容将会被纳入编译单元

![ifndef验证.png](static/ifndef验证.png)
验证下：将头文件内容重复复制到math函数中，会发现第一次内容正常，第二次的内容置灰了，因为log.h定义过了，这个是头文件保护机制

### include<>和include""区别
<>指的是从include指定目录去拉去文件引入
""指的是从当前的相对目录去引入文件
当然""同样可以在编译器指定目录去引入文件 但编译器指定目录引用通常用<>
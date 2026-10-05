### 类或者结构体外部使用static
仅在当前的编译单元可使用
这个在之前编译链接将结果，当有多个

在其中一个文件撰写函数
```cpp
int s_Variable = 9;
```
另一个函数同样有s_Variable这个变量
```cpp
#include <iostream>

int s_Variable = 9;
int main()
{
    std::cin.get();
}
```
build发现报错了
![定义多个变量报错.png](static/定义多个变量报错.png)
增加static修饰
```cpp
static int s_Variable = 9;
```
重新build不报错
### 类或者结构体内部使用static
修饰内部变量和方法的时候，被类的所有实例共享同一个静态变量


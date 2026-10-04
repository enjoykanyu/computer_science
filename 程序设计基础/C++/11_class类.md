类是面向对象编程中过重要组成部分

像Java属于面向编程语言 程序的撰写和思想都符合面向对象编程 C语言不支持面向对象编程 C++既支持面向对象编程同时支持面向过程编程

# 类
### 举例
```cpp
#include <iostream>

class Player{

    int x,y;
    int speed;
};
int main()
{
    Player player;
    player.x = 9;
    std::cin.get();
}
```
像这样和Java差不多定义一个类和成员变量
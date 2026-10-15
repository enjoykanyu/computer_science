栈（Stack）本质上是一个（**后进先出LIFO - Last In First Out**）的数据结构。

每当我们进入一个新的作用域（Scope），就相当于向书堆顶端新放了一本书；在该作用域内声明的所有局部变量，就像是在这本书上写字。
一旦作用域结束，这本书被移走（弹出退栈），上面记录的所有局部变量和对象将当场被全数销毁，占用的内存立刻释放。

### 构造函数创建析构函数销毁
```cpp
#include <iostream>

class Entity {
public:
    Entity() {
        std::cout << "Created Entity" << std::endl;
    }
    ~Entity() {
        std::cout << "Destroyed Entity" << std::endl;
    }
};
```

### 栈分配
```cpp
int main() {
    {
        Entity e; // 栈上分配
    } // 离开花括号，离开作用域
    std::cin.get();
}
```

### 堆分配
```cpp
int main() {
    {
        Entity* e = new Entity(); // 堆上分配
    } // 离开作用域，仅指针变量 e 本身销毁，所指堆内存未释放！
    std::cin.get();
}
```
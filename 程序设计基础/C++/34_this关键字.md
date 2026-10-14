- 定义：this 是 C++ 中的一个核心关键字，它只能在类的非静态成员函数（Member Functions）内部访问。
- 本质：它是一个指针，指向当前调用该方法的对象实例（Current Object Instance）。
- 为什么需要它？ 非静态方法必须由一个具体的有效对象来触发调用，而成员函数在执行时，就是通过 this 指针来确认自己到底在操作哪一块内存中的数据


编译器在编译时，会默默地把 this 指针作为函数的第一个隐藏形参塞进去。

自己写的函数
```cpp
void Entity::Print();
e.Print();
```
实际

```cpp
void Entity_Print(Entity* const this); // 隐藏的第一个参数
Entity_Print(&e);                       // 将当前对象地址传给 this（通常放在 ECX 寄存器或 RDI 寄存器中）
```

### 变量重名消歧义
```cpp
class Entity {
public:
    int x, y;

    Entity(int x, int y) {
        // 如果想在构造函数体内赋值：
        x = x; // ❌ 编译可以通过，但毫无意义！只是局部形参 x 给自己赋值
    }
};
```
借助this
this 是指向当前实例的指针，所以可以通过指针访问成员：
```cpp
this->x = x;
this->y = y;
```
解引用用法
```cpp
(*this).x = x;
```

![this底层.png](static/this底层.png)
在非 const 成员函数中，如果你把鼠标悬停在 this 上，它的完整类型其实是：
```cpp
Entity* const this; // 顶层 const（Top-level const）
```

```cpp
- this = nullptr; // ❌ 编译直接报错！不能给 this 赋新地址
```

### 只读函数this
```cpp
class Entity {
public:
    int x, y;

    int GetX() const {
        // 这里的 this 究竟是什么类型？
        // 悬停会发现是：const Entity* const
        return this->x;
        // return x; // 等价于 return this->x;
    }
};
```
![const函数this为const修饰类.png](static/const函数this为const修饰类.png)
当this->x=9会报错
![const函数this-%3Ex禁止修改数据.png](const函数this-%3Ex禁止修改数据.png)
因为是指向 const Entity 的指针，所以只要你尝试在函数内写 this->x = 5;，编译器直接判定为非法修改常量

### 在类内部把自己传递给外部函数
```cpp
class Entity {
public:
    int x, y;

    void Print() const {
        // 如何把“当前对象”传给外部函数？
        PrintEntity(*this); //  对 this 指针解引用！
    }
};

void PrintEntity(const Entity& e) {
    // 具体的打印逻辑...
}
```
- 对指针进行解引用 *this，即可得到当前对象的实体/引用（Reference）；
- 如果当前处于非 const 成员函数中，*this 得到的是 Entity&（普通引用）；
- 如果当前处于 const 成员函数中，*this 得到的是 const Entity&（常量引用），完美匹配外部函数的接收参数。

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
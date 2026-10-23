C++数组本该叫 ArrayList 或者 DynamicArray。把它叫成 vector 是历史命名失误。
容器与模板库（STL）：
std::vector 是 C++ 标准模板库（STL）中被使用最频繁的容器。它本质上就是一个支持自动动态扩容的数组。
与传统原生数组的对比：
- 传统栈数组 int arr[5] 或堆数组 new int[5] 大小在运行时不可改变，一旦数据超过容量就会越界崩溃。
- std::vector 允许你在创建时完全不知道未来会有多少数据，随着你不断往里添加元素，它会在底层自动完成扩容

### vector是如何实现动态扩容的
当前容器的容量（Capacity）不足以容纳新插入的元素时，vector 在幕后执行了极其硬核的搬迁操作：
1. 申请更大内存：在堆上重新分配一块比原空间更大的连续内存块（通常扩容为原来的 1.5 倍或 2 倍）；
2. 迁移老数据：将旧数组里的所有已有元素**全数拷贝（或移动）**到这块新内存中；
3. 释放旧内存：依次析构旧数组中的对象，并释放（delete）掉旧的堆内存；
4. 追加新数据：将新加入的元素写入新内存空间末尾


### 容量（Capacity）和大小（Size）的区别：
- size()：当前数组中实际存放了多少个有效元素；
- capacity()：当前底层堆内存最多能容纳多少个元素而无需再次重新分配

```cpp
#include <iostream>
#include <vector>

struct Vertex {
    float x, y, z;
};

// 重载输出运算符方便打印
std::ostream& operator<<(std::ostream& stream, const Vertex& vertex) {
    stream << vertex.x << ", " << vertex.y << ", " << vertex.z;
    return stream;
}
```
在 vector 中，到底应该存对象本身还是存对象的指针？
即：std::vector<Vertex> vs std::vector<Vertex*>？

选择 std::vector<Vertex>（直接存对象，强烈推荐首选）：
- 内存连续性（Contiguous Memory）：所有 Vertex 在内存中一个接一个紧密排列（xyz xyz xyz...）。
- CPU 缓存行友好（Cache Friendly）：现代 CPU 每次从内存读取数据都是以 Cache Line（通常 64 字节）为单位。内存连续意味着 CPU 一次缓存命中就能读取多个相邻顶点，遍历读取性能极其恐怖！

选择 std::vector<Vertex*>（存堆指针）：
- 指针本身是连续的，但指针所指向的实际数据分散在堆内存的不同角落（内存碎片化）。CPU 在遍历时会不断遭遇缓存不命中（Cache Miss），频繁向慢速主存寻址，速度大幅暴跌。

指针唯一的优势： 如果存储的对象体积极其庞大且拷贝开销极高，扩容搬家时拷贝 8 字节的指针确实比拷贝大对象更快。

黄金准则： 只要条件允许，永远默认使用 std::vector<T> 直接存储对象，指针仅作特殊万不得已的备选



### API
添加元素：push_back
```cpp
std::vector<Vertex> vertices;
// C++11 统一列表初始化：
vertices.push_back({ 1, 2, 3 });
vertices.push_back({ 4, 5, 6 });
```
遍历元素：下标法 vs 基于范围的 for 循环
- 传统下标法：
```cpp
for (size_t i = 0; i < vertices.size(); i++) {
    std::cout << vertices[i] << std::endl; // 重载了 operator[]，支持随机访问
}
```
- 基于范围的 for 循环与致命拷贝陷阱：
```cpp
// 错误示范：按值传递，每一个元素都会在遍历时触发一次完整的拷贝构造！
for (Vertex v : vertices)
    std::cout << v << std::endl;

//  正确做法：必须带上引用符号 &，只读时加上 const！
for (const Vertex& v : vertices)
    std::cout << v << std::endl;
```
元素移除：clear 与 erase
- 全量清空：vertices.clear(); 将有效大小（Size）置为 0，但底层的连续内存（Capacity）并不会释放给操作系统。
- 按位置删除（erase 与迭代器 Iterator）：
erase 不能直接传下标数字（如不能写 erase(1)），它强制接收一个迭代器（Iterator）
```cpp
// 删除第 2 个元素（索引为 1 的位置）：
vertices.erase(vertices.begin() + 1);
```
本质：vertices.begin() 是指向首元素的迭代器，通过 + 1 偏移到第 2 个元素位置完成移除

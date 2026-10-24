当 std::vector 的容量不足时，会发生重新分配（Reallocation）：
1. 申请更大堆内存；
2. 将老数据逐个拷贝过去；
3. 释放旧内存。
   这个过程不仅伴随着昂贵的堆内存申请/释放，更伴随着大量已有对象的深拷贝

因此需考虑vector优化   
```cpp
#include <iostream>
#include <vector>

struct Vertex {
    float x, y, z;

    // 带参构造函数
    Vertex(float x, float y, float z)
        : x(x), y(y), z(z) {}

    // 拷贝构造函数（作为探针）
    Vertex(const Vertex& other)
        : x(other.x), y(other.y), z(other.z) {
        std::cout << "Copied" << std::endl;
    }
};
```

在 main 函数中连续 push_back 三个顶点：
```cpp
int main() {
    std::vector<Vertex> vertices;
    vertices.push_back(Vertex(1, 2, 3));
    vertices.push_back(Vertex(4, 5, 6));
    vertices.push_back(Vertex(7, 8, 9));

    std::cin.get();
}
```
![6次拷贝输出.png](static/6次拷贝输出.png)
输出了6次 不符合预期

### 多次输出拷贝的原因
  执行第 1 行：vertices.push_back(Vertex(1, 2, 3));
- 现象：输出 1 次 Copied。
- 原因（原因 A：传参拷贝）：Vertex(1, 2, 3) 首先在 main 函数当前的栈帧中被临时构造出来；为了存进 vector，必须把它从当前栈内存拷贝一份放到 vector 内部的堆内存中。
- 此时 size = 1, capacity = 1。

执行第 2 行：vertices.push_back(Vertex(4, 5, 6));
- 现象：输出了 2 次 Copied（累计 3 次）。
- 原因：
   1. 传参拷贝：新元素 (4, 5, 6) 从栈拷贝进 vector（第 1 次）；
   2. 扩容迁移（原因 B）：此时原有容量为 1 已满，vector 扩容到容量 2，不得不把老元素 (1, 2, 3) 从旧内存深拷贝到新内存（第 2 次）！
- 此时 size = 2, capacity = 2。

执行第 3 行：vertices.push_back(Vertex(7, 8, 9));
- 现象：输出了 3 次 Copied（累计 6 次）！
- 原因：
   1. 传参拷贝：新元素 (7, 8, 9) 从栈拷贝进 vector（第 1 次）；
   2. 再次扩容迁移（原因 B）：容量 2 再次耗尽，vector 扩容到 3（或更高），把之前的两个老元素 (1, 2, 3) 和 (4, 5, 6) 重新全部深拷贝一次到新内存（第 2 次和第 3 次）
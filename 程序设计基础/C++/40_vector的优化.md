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
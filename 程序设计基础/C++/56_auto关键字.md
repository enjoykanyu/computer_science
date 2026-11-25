### auto的基本含义
显式声明：
```cpp

int a = 5;
int b = a;
```
使用 auto：
```cpp
auto a = 5;
auto b = a;
```
编译器会根据初始化表达式推导类型：
```cpp
auto a = 5;   // int
auto b = a;   // int
```
因此 auto 必须有初始化值：
```cpp
auto value;   // 错误，无法推导类型
```
正确
```cpp

auto value = 10;
```
从 C++ 标准语义看，auto 是占位类型说明符。编译器会根据初始化器执行模板参数推导规则，推导出一个确定类型，然后把变量当作普通静态类型变量处理。

所以：
```cpp
auto value = 10;
```
不是运行时动态类型，而大致等价于：
```cpp
int value = 10;
```
### 字面量的影响
```cpp
auto a = 5;      // int
auto b = 5L;     // long
auto c = 5.5f;   // float
auto d = 5.5;    // double
auto e = "Cherno"; // const char*
```

```cpp
auto text = "Cherno";
```
推导出来不是 std::string，而是const char*

需要标准字符串：
```cpp
auto text = std::string("Cherno");
// 推导为 std::string
或者
std::string text = "Cherno";
```

### auto不会让C++变成弱类型语言
代码中不再显式写出类型：
```cpp
auto value = 5;
```
但 value 仍然具有确定类型：
```cpp
value = 10;       // 合法
value = "Hello";  // 错误，不能把字符串赋给 int
```
因此 auto 只是省略类型名称，不能改变 C++ 的静态类型系统。

下面代码不是“同一个变量类型动态改变”：
```cpp
auto value = 5;
```
它只能是 int。如果改成：
```cpp
auto value = "Hello";
```
这是重新编译后生成了另一个 const char* 变量
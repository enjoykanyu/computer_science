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


### auto的优点：修改初始化表达式时不必修改左侧类型
```cpp
auto value = 5;
```
改成
```cpp
auto value = 5.5f;
```
左侧仍然是
```cpp
auto value
```
只不过推导结果由 int 变成了 float。

这可以减少代码修改量，但也带来风险：右侧表达式类型改变后，左侧变量类型会悄悄改变。

例如：
```cpp
auto value = GetValue();
```
GetValue() 的返回类型从 int 改为 double，value 也会自动从 int 变为 double。

这可能导致：
- sizeof(value) 结果变化；
- 序列化格式变化；
- 重载函数选择变化；
- 精度和性能变化；
- 后续代码行为改变。
  所以 auto 带来的便利与隐藏变化是同时存在的
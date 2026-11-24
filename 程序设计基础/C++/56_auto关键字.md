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
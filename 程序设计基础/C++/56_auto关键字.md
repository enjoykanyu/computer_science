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
  



### auto的两面性
```cpp
std::string GetName()
{
    return "Cherno";
}
std::string name = GetName();
auto name = GetName();
```

name 的类型是 std::string
两种写法功能相同

API 返回类型发生变化
将函数改为
```cpp
const char* GetName()
{
    return "Cherno";
}
```
则
```cpp
auto name = GetName();
```
推导为
```cpp
const char* name;
```
此时下面代码不再成立：
```cpp
name.size(); // 错误，const char* 没有 size()
```

写成：
```cpp
std::string name = GetName();
```
则由于 std::string 可以从 C 字符串构造，仍然可能正常工作：
```cpp
std::string name = GetName();
name.size(); // 合法
```
体现了auto的两面：
- auto的优点：API 返回类型变化时，调用处不需要同步修改：
```cpp
auto name = GetName();
```
auto 的风险
调用者可能没有意识到变量类型已经改变，导致后续行为变化。

例如：
```cpp
auto name = GetName();
std::cout << sizeof(name) << std::endl;
```
如果返回 std::string，sizeof(name) 是字符串对象大小；如果返回 const char*，则通常是指针大小。这种变化可能破坏二进制协议、序列化或内存布局。

因此，若变量必须明确表示某种语义类型，建议写：
```cpp
std::string name = GetName();
```
而不是
```cpp
auto name = GetName();
```

### 不建议在简单类型上滥用auto
不建议这样写
```cpp
auto count = 10;
auto name = std::string("Cherno");
auto speed = 3.5f;
```

这些类型从初始化表达式中很容易看出，写成显式类型通常更清楚：
```cpp
int count = 10;
std::string name = "Cherno";
float speed = 3.5f;
```
因为读者可以直接知道：
```cpp
count 是 int
name 是 std::string
speed 是 float
```
而看到
```cpp
auto name = GetName();
```
读者必须：
- 查看 IDE 提示；
- 跳转到函数定义；
- 查看头文件声明；
- 分析模板返回类型
![img_7.png](img_7.png)
变量名称大多数相同，Java的布尔变量是boolean C++是 bool
# 编程范式
Java全都是在一个类中撰写程序，C++则是混合了C的面向过程和面向对象的混合编程范式
# 编译顺序
![img_8.png](img_8.png)
Java中编写函数A B C A调用B B调用C 无论它的前后顺序怎样 Java编译过程中都不会出错（Java编译器会浏览全部内容再回溯获取整体结构）
但C++会从上往下一次编译完成，假设GetScoresAndAverage函数注释掉或者写在main函数的下边，它就会报错
![img_9.png](img_9.png)
编译
![img_10.png](img_10.png)
可以看到编译不通过

```cpp
double test333 (double a, double b);
int main () {

  test333(3,3);
}
double test333 (double a, double b) {
  return (a + b) / 99;
}
```

改成这样就行了
在这之前声明下原型（作用提前告诉编译器函数会在下边出现）
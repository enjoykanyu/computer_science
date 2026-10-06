枚举类不设置初始值则会从0开始计算递增，若设置了第一个变量 后续都会递增加一
```cpp
enum{
    A,B,C
};

int main(int argc, char const *argv[])
{
    if (A==1)
    {
        /* code */
    }
    
    return 0;
}

```
![img_20.png](img_20.png)
可以看到C被递增赋予了2

类型可以设置枚举存储类型unsigned char 占1个字节，同样可以不设置
```cpp
enum Value:unsigned char
{
    A,B,C
};

int main(int argc, char const *argv[])
{
    if (A==1)
    {
        /* code */
    }
    
    return 0;
}

```
枚举只可以存储整数
![img_21.png](img_21.png)
当设置为float可以看到编译报错了


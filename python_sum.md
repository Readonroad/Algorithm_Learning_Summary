## python 基础
### 函数的参数
定义函数时，把参数的名字和位置确定下来，函数接口定义就完成了。函数调用时，只需要知道如何传入正确的参数，以及函数将返回什么样的值就可以了。
#### 位置参数
位置参数，调用时，按照位置顺序传入对应的值
```python
def power(x,n):
    return x**n
#调用
power(5,2)
```
#### 默认参数
为函数定义的参数设定默认值，调用时，不传入改参数值，即使用默认值。默认参数要在必选参数之后。调用时，既可以安全顺序提供默认值，也可以不按照顺序提供默认值，此时需要把参数名写上。默认参数必须是不可变兑现
```python
def power(x,n=2):
    return x**n
#调用
power(5) 等价于 power(5,2) 或 power(5,n=2)
```
#### 可变参数
传入的参数个数是可变的。
```python
#方式1，将可变参数作为list或tuple传入
def calc(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
calc([1,2,3])  或 calc((1,2,3,4))

#方式2,函数定义为可变参数,在参数前面加*号，函数内部接收到的是一个tuple
def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
calc(1,2,3,4)
#可以通过在list前面加上*号，将list所有元素作为可变参数传入
nums = [1,2,3,4]  calc(*nums)
```
#### 关键字参数
允许传入0或任意个**含参数名**的参数，这些关键字参数在函数内部自动组装为一个dict。好处，可以扩展函数的功能，如果调用者愿意提供更多的参数，也可以接收。
```python
def person(name,age,**kw):
    print('name:',name,'age:',age, 'other:',kw)
person('Anna',10)   #只传入必选参数
person('Bob', 35, city='Beijing')  #传入一个关键字参数，或多个

#将字典转化为关键字参数传入，在变量名前加 **
extra = {'city': 'Beijing', 'job': 'Engineer'}
person('Jack', 24, **extra)
```
#### 命名关键字参数
关键字参数，调用者可以传入任意不受限制的关键字参数，对于传入了那些，需要在函数内部通过kw检查
```python
def person(name,age,**kw):
    if 'city' in kw:
        ...
    if 'job' in kw:
        ...
    print('name:',name,'age:',age, 'other:',kw)
```
如果要限制关键字参数的名字，可以用命名关键字参数。命名关键字参数需要一个特殊分隔符*，*后面的参数被视为命名关键字参数。
```python
#限制只接受city和job作为关键字参数
def person(name,age,*,city,job):
    print('name:',name,'age:',age, 'city:',city,'job':job)
#调用时，需要传入关键字参数名，如果不传入，会报错。命名关键字可以有默认值
person('Jack', 24, city='Beijing', job='Engineer')
```
如果函数中已经有了一个可变参数，后面跟着的命名关键字参数就不需要一个特殊分隔符 *
```python
#限制只接受city和job作为关键字参数, *args是可变参数
def person(name, age, *args, city, job):
    print(name, age, args, city, job)
```
#### 函数参数组合
在Python中定义函数，可以用必选参数、默认参数、可变参数、关键字参数和命名关键字参数，这5种参数都可以组合使用。参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数。

### python迭代
给定一个list或tuple，可以通过for循环来遍历，即迭代。
```
dict类型的迭代，默认是key
for key in dict:             #默认是key值迭代
    print(key)
for value in dict.values():  #对值迭代，dict.values();
    print(value)
for key,value in dict.items(): #对元素迭代，dict.items()
    print(key,value)
list类型可以通过enumerate函数变成索引-元素对
for i,value in enumerate(list):
```
判断一个对象是否为可迭代对象，通过collections模块的iterable类型判断
```python
from collections import Iterable
isinstance(a, Iterable)  #判断变量a是否是可迭代对象，通过内置函数isinstance判断是否为某种类型
```
### 列表生成器VS生成器
```python
#通过列表生成器可以直接生成一个列表，受内存限制，列表容量有限。元素很多时，会占用很大的存储空间
ls = [ x*x for x in range(10)]

#通过一边循环一边计算的机制，成为生成器。生成器保存的是算法，每次调用next(g),就可以计算下个元素的值。迭代器也是可迭代对象，也可以用for循环访问
g = (x*x for x in range(10))
#获取生成器的元素
next(g)
for i in g:
    print(i)

#通过定义函数生成generator
def fib(max):
    n,a,b = 0,0,1
    while n < max:
        yield b    #通过yield关键字，可以将函数变成generator，每次调用next()函数时，遇到yield语句返回，再次执行时，从上次返回的yield语句处继续执行
        a,b = b,a+b
        n=n+1
    return 'done'
```
### 迭代器
Python中可用于for循环的对象统称为可迭代对象，即Iterable。通过内置函数isinstance(a,Iterable)判断一个对象是否是可迭代对象
```
集合数据类型，如list,tuple,dict,set,str等
generator,包括生成器和带yield的generator function
```
生成器还可以通过next()函数不断调用返回下一个值。可以被next()函数调用并不断返回下一个值得对象称为迭代器，即Iterator

### 函数式编程
函数式编程就是一种抽象程度很高的编程范式，纯粹的函数式编程语言编写的函数没有变量，因此，任意一个函数，只要输入是确定的，输出就是确定的。函数式编程的一个特点就是，允许把函数本身作为参数传入另一个函数，还允许返回一个函数。
#### 高阶函数
变量可以指向函数，函数的参数能接收变量，那么一个函数就可以接收另一个函数作为参数，这种函数就称之为高阶函数。  能把函数作为参数传入，这样的函数即为高阶函数
```python
#求绝对值abs函数
f = abs    #变量可以指向函数
def add(x,y,f) : # f作为add的参数，则add函数为高阶函数
    return f(x)+f(y)
```
##### map/reduce
map()函数接收两个参数，一个是函数，一个是Iterable, map将传入的函数依次作用到Iterable中的每个元素上，并把结果作为新的Iterator返回。
```python
def f(x):
    return x*x
r = map(f, [1,2,3,4])
list(r)
```
reduce()把一个函数作用在一个可迭代对象上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积运算
```python
from functools import reduce
def fn(x,y):
    return x*10+y
reduce(fn, [1,2,3,4])
#返回结果
1234
```
#### filter 过滤序列
filter()也接收一个函数和一个序列。filter把传入的函数依次作用于每个元素，根据返回值是True和False决定是否保留该元素。
```python
def is_odd(x)
    return x%2 == 1
list(filter(is_odd,[1,2,3,4,5]))
```
#### sorted函数
```python 
#几首一个key函数来实现自定义的排序
sorted([1,2,-1,3,5],key =abs)  #按绝对值大小进行排序
sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower,reverse=True)  #忽略大小写的排序吗，reverse为True，反向排序
```

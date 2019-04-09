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
### 函数作为返回值
高阶函数可以接受函数作为参数，还可以把函数作为结果值返回
```python
def calc_sum(*args):
    ax = 0
    for n in args:
        ax+=n
    return ax

#如果不需要立刻求和，则可以不返回求和的结果，而返回求和的函数
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:  
            ax+=n
        return ax
    return sum

#当调用lazy_sum()函数时，返回的不是求和结果，而是求和函数
#当调用lazy_sum()函数时，每次调用都会返回一个新的函数，即使传入的参数相同
f=lazy_sum([1,2,3,4])
#调用函数f后，才能计算求和的结果
f()   #执行结果为10
```
返回的函数在其定义内部引用了局部变量args，所以，当一个函数返回了一个函数后，其内部的局部变量还被新函数引用。
```python 
def count():
    fs=[]
    for i in range(1,4):
        def f():
            return i*i
        fs.append(f)     #将函数作为返回值加入fs中，
    return fs

f1,f2,f3 = count()  
#执行结果
f1(),f2(),f3()   #执行结果均为9，返回的函数引用了变量i,但并未立即执行，等三个函数都返回时，引用的变量i变成了3，故最终结果为9
```
返回函数最好不要引用任何循环变量，或者后续会发生变化的变量;如果要引用循环变量，就再创建一个函数，用改函数的参数绑定循环变量当前的值
```python
def count():
    def f(j):
        def g():
            return j*j
        return g
    fs=[]
    for i in range(1,4):
        fs.append(f(i))     #将函数作为返回值加入fs中，
    return fs

f1,f2,f3 = count() 
f1(),f2(),f3()    #执行结果为1,4,9
```
### 装饰器
函数是一个对象，函数对象可以被赋值为变量，通过变量也能调用该函数
```python
def now():
    print('2019-4-4')

f=now   #函数赋值给变量
f()     #通过变量调用函数，等价于now()
#函数对象属性 __name__,可以获取函数的名字
now.__name__   # ’now'
```
若希望增加now()函数的功能，如在调用前后自动打印日志，又不希望修改now()函数的定义，这种在代码运行期间动态增加功能的方式，称之为“装饰器”。decorator是一个返回函数的高阶函数。
```python
import functools
def log(func):
    def wrapper(*args,**kw):
        @functools.wraps(func)    #保留原函数 func 的属性
        print("call %s():" %func.__name__)
        return func(*args,**kw)
    return wrapper

#在now()函数定义出，增加 @log,执行now()的时候，相当于执行语句：now=log(now)
@log
def now():
   print('2019-4-4')

#调用now
now()  
#结果为
#call now():
#2019-4-4
```
如果decorator本身需要传入参数，这需要编写一个返回decorator的高阶函数
```python 
import functools
def log(text):
    def decorator(func):
        @functools.wraps(func)   #保留原函数 func 的属性
        def wrapper(*args,**kw):
            print('%s %s():'%(text,func.__name__))
            return func(*args,**kw)
        return wrapper
    return decorator

#定义前，增加@log('excute')  相当于执行语句：now=log('excute')(now)
@log('excute')
def now():
    print('2019-4-4')

#执行
now()
#结果为
#excute now():
#2019-4-4
now.__name__   #结果为 'wrapper'
```
先执行log('execute')，返回的是decorator函数，再调用返回的函数，参数是now函数，返回值最终是wrapper函数。经过decorator装饰后，函数的属性__name__变成'wrapper'，要想保留原来函数的属性，就需要将原始函数的__name__等属性复制到wrapper函数中。利用python内置的functools.wraps。

### 偏函数
functools.partial 帮助创建一个偏函数，作用是把一个函数的某些参数固定住（即设置默认值），返回一个新的函数，调用这个新的函数会更加简单。
```python
import functools
int2 = functools.partial(int,base = 2)    #int()将输入按给定的base转化为对应的整数
```
## python面向对象
类是创建实例的模板，实例是一个一个具体的对象,各个实例拥有的数据相互独立，互不影响。方法是与实例绑定的函数，与普通函数的区别在于，方法可以直接访问实例的数据。

```python
#类的好处封装数据
class Student(object):

    def __init__(self, name, score):  #特殊方法，前后用两个下划线
        self.name = name
        self.score = score
    
    def printStudent(self):
        print('%s %s' % (self.name, self.score))
```
类中定义的函数，与普通函数相比，第一个参数永远是实例变量self，self指向创建的实例本身，函数调用时，不用传递改参数，除此之外，类的方法跟普通函数没有区别。
### 访问权限
变量名前加两个下划线__,就变成了一个私有变量，只有内部可以访问，外部不可以访问。私有变量可以访问外部代码随意修改对象内部状态，加入访问权限的保护，代码更加健壮
```python
class Student(object):

    def __init__(self, name, score):  #name和score为私有变量
        self.__name = name
        self.__score = score

    def print_score(self):
        print('%s: %s' % (self.__name, self.__score))

 bart = Student('Bart Simpson', 59)
 bart.__name    #访问错误，__name是私有变量，不能访问
```
外部代码如果要获取和修改数据，可以在对应的类中增加获取变量和修改变量的方法。
```python 
class Student(object):
    ...
    def get_name(self):
        return self.__name

    def get_score(self):
        return self.__score

    def set_name(self,name):
        self.__name = name

    def set_score(self,score):
        self.__score = score
```
通过在类中定义方法修改类变量的好处是，在方法中，可以对参数做检查，避免传入无效的参数。当变量不是私有变量时，也可以通过self.name赋值的方式修改变量。但是这种方法会传入无效的参数。
```
* 私有变量被python解析器改成了另一个变量名，如果知道修改后的变量名，就可以对该变量进行访问。
* 一个下划线开头的变量名不是私有变量，外部可以访问，但是按照约定俗成的规定，当你看到这样的变量时，意思就是，“虽然我可以被访问，但是，请把我视为私有变量，不要随意访问”。
* 通过 Student bart.__name = 'xxx' 的方式可以设置成功，但是并不是修改了变量名，而是为改实例新增了一个__name的变量。因为__name变量和class内部的__name变量不是一个变量！内部的__name变量已经被Python解释器自动改成了另一个变量名。
* python没有机制去阻止不好的行为，因此要按照规范写
```
### 继承
继承最大的好处是子类可以获得父类的全部功能。
```python
class Animal(object):
    def run(self):
        print('Animal is running...')

#Cat继承Animal，可以拥有父类的run方法
class Cat(Animal):
    pass    
#Dog继承Animal
class Dog(Animal):

    def run(self):   
        print('Dog is running...')

    def eat(self):
        print('Eating meat...')
```
当子类父类存在相同的方法时，子类的方法覆盖了父类的方法，代码运行时，会调用子类的方法。即继承的另一个好处，即为多态。
#### 多态
如果一个类继承另一个类，那么子类变量的数据类型是该子类，同时也是父类的数据类型。但是父类类型的变量并不是子类类型变量。
```python
#定义一个参数为父类类型的函数
def run_twice(animal):
    animal.run()
    animla.run()

run_twice(Animal())
#结果：
# Animal is running...
# Animal is running...
rum_twice(Dog())
#结果：
# Dog is running...
# Dog is running...
```
多态的好处在于当需要传入子类变量时，可以只接受父类类型。因为子类继承父类，传入的变量无论是子类类型还是父类类型，都会自动调用实际类型的方法。同时，当新增一个继承父类的子类时，只需要关注子类中的方法编写正确，而不需要关注代码的调用。
### 获取对象类型
```
type()  判断对象类型，基本数据类型都可以用type()判断,返回对应的Class类型
函数类型可以使用types模块中定义的常量判断如，types.FunctionType、types.BuiltinFunctionType、types.LambdaType、types.GeneratorType
instance() 使用更广泛，对继承关系也很实用，优先使用该函数判断
```
### 面向对象进阶
#### __slots__
定义一个Class，创建该类的实例，可以给该实例绑定任何属性和方法。给一个实例绑定属性和方法，只对该实例生效，对其他实例无用。
```python 
class Student(obejct):
    ...
st = Student()
st.name = 'John'   #动态绑定name属性

#定义一个函数
def set_age(self, age):
    self.age = age

#将定义的方法绑定到实例st中   
from types import MethodType
st.set_age = MethodType(set_age, st)
st.set_age(25)
```
如果要对所有的实例都绑定方法，可以对class绑定方法
```python 
class Student(obejct):
    ...
st = Student()
st.name = 'John'   #动态绑定name属性

#定义一个函数
def set_age(self, age):
    self.age = age
#对Student类绑定方法，所有实例均可调用
Student.set_age = set_age
```
如果需要限制实例绑定的数据，可以在class的定义中，定义一个特殊的__slot__变量，限制类实例添加的属性。__slots__定义的属性仅对当前类实例起作用，对继承的子类不起作用。如果子类中定义了__slots__，那么子类实例允许定义的属性就是自身的__slots__加上父类的__slots__
```python
class Student(object):
    __slots__ = ('name', 'age')   #用tuple定义允许绑定的属性名称,没在该定义中，会报错
```
#### @property
实例绑定属性时，如果将属性直接暴露出去，导致没有办法检查参数
```python
st = Student()
st.score = 999 #该属性值是无效值
#可以通过在Class Student中定义get 和 set score的函数来对score的赋值进行校验
```
```python
class Student(object):
    def get_score(self):
        return self._score

    #对score的值进行校验
    def set_score(self, value):
        if not isinstance(value,int):
            raise ValueError('score must be an integer.')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100')
        self._score = value
```
通过get和set的方法比较复杂，通过@property可以实现既能对score进行参数检查，也可以用类似属性的方式访问类变量
```python
class Student(object):
    @property
    def score(self):
        return self._score    #通过@property将score的getter方法变成属性
    
    @score.setter       #通过另个一个装饰器@score.setter，将setter方法变成属性赋值
    def score(self, value):
        if not isinstance(value,int):
            raise ValueError('score must be an integer.')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100')
        self._score = value
```
用@property只定义getter方法，不定义setter方法的是一个只读属性。

### 定制类
前两两个下划线的变量或函数名，在python中是有特殊用途的。
```
__len__()       返回长度的方法
__str__()       打印字符串的方法
__iter__()      一个类想被用于for...in 循环，需要实现__iter__()方法，返回一个迭代对象，然后for循环会不断调用该迭代对象的__next__()方法拿到循环的下一个值，直到终止
__getitem__()   根据下表取出对应的元素
__getattr__()   动态返回一个属性
__call__()      直接对实例进行调用
```
```python
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1 # 初始化两个计数器a，b

    def __iter__(self):
        return self # 实例本身就是迭代对象，故返回自己

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b # 计算下一个值
        if self.a > 100000: # 退出循环的条件
            raise StopIteration()
        return self.a # 返回下一个值
    
    #获取元素或切片元素
    def __getitem__(self, n):
        if isinstance(n, int): # n是索引
            a, b = 1, 1
            for x in range(n):
                a, b = b, a + b
            return a
        if isinstance(n, slice): # n是切片
            start = n.start
            stop = n.stop
            if start is None:
                start = 0
            a, b = 1, 1
            L = []
            for x in range(stop):
                if x >= start:
                    L.append(a)
                a, b = b, a + b
            return L
```
## 异常、调试
```
异常语法try ... catch ... finally
调试方式：通过打印函数和断言辅助(print/assert) / 记日志(logging) /pdb调试
```
## 序列化
python中提供了pickle模块实现序列化，把变量从内存中变成可存储或传输的过程称为序列化，称为pickling，序列化后，可以把序列化后的内容写入磁盘，或者通过网络传输到别的机器上。反过来，把变量内容从序列化的对象重新读到内存里称为反序列化unpickling
```python
import pickle
d = xxx
pickle.dumps(d)  #dumps()将任意对象序列化成一个bytes，然后存储

f = open('dump.txt','wb')
pickle.dump(d,f) #对象序列化后写入一个file-like Object,并存储到文件f中

d_new = pickle.load(f)  #将数据还原，用load()方法从一个file-like Object钟直接反序列化出对象  或 pickle.loads(f)
f.close()  
```
## 多进程
Unix/Linux操作系统提供了一个fork()系统调用，它非常特殊。普通的函数调用，调用一次，返回一次，但是fork()调用一次，返回两次，因为操作系统自动把当前进程（称为父进程）复制了一份（称为子进程），然后，分别在父进程和子进程内返回。子进程永远返回0，而父进程返回子进程的ID。因为一个父进程可以fork出很多子进程，故父进程要记下每个子进程的ID,子进程通过调用getppid()可以拿到父进程的ID。

python 的os模块封装了常见的系统调用，包括fork，通过os.fork()可以创建子进程

windows没有fork调用，windows平台上，可以使用multiprocessing模块。

### multiprocessing模块
multiprocessing模块是跨平台版本的多进程模块，该模块提供了一个Process类来代表一个进程对象。
```python
from multiprocessing import Process
import os

#子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' %(name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')
```
Process创建子进程，只需要传入一个执行函数和函数的参数。start()启动函数。join()方法可以等待子进程结束后再继续向下执行，通常用于进程间同步
### 进程池pool
如果需要启动大量子进程，可以用进程池的方式批量创建子进程
```python
from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)      #Pool()线程池
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
```
pool对象调用join()方法会等待所有子进程执行完毕，调用join之前必须先调用close()，调用close()之后就不能继续添加新的Process了。Pool的默认大小是CPU的核数
### 进程通信
Python的 multiprocessing 模块包装了底层的机制，提供了Queue、Pipes等多种方式来交换数据，从而实现进程间的通信。
```python
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()
```
在Unix/Linux下，可以使用fork()调用实现多进程。

要实现跨平台的多进程，可以使用multiprocessing模块。

进程间通信是通过Queue、Pipes等实现的。

## 多线程
多任务可以由多进程完成，也可以由一个进程内的多线程完成。python中的threading模块可以实现多线程。启动一个线程就是把一个函数传入并创建Thread实例，然后调用start()开始执行。
```python
import time,threading

#新线程执行的代码
def loop():
    print('thread %s is runing...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n=n+1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)

print('thread %s is running...' % threading.current_thread().name)
t = threading.Thread(target=loop, name='LoopThread')  #新线程，并指定新线程的名字
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)
```
任何一个进程默认会启动䘝线程，该线程称为主线程，主线程可以启动新的线程。threading模块有个current_thread()函数，返回当前前程的实例，主线程实例的名字叫MainThread,子线程的名字在创建时指定，如果不指定名字，则会由python自动命名。

### lock
多线程和多进程最大的不同在于，多进程中，同一个变量，各自有一份拷贝存在于每个进程中，互不影响，而多线程中，所有变量都由所有线程共享，所以，任何一个变量都可以被任何一个线程修改，因此，线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。
```python
import time, threading

# 假定这是你的银行存款:
balance = 0

def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n
def run_thread(n):
    for i in range(100000):
        change_it(n)

#加锁后的实现
def run_thread_lock(n):
    for i in range(100000):
        lock.acquire()     #获取锁
        try:
            change_it(n)
        finally:
            lock.release()  #释放锁
t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
```
### ThreadLocal
多线程环境下，每个线程都有自己的额数据，一个线程使用自己的局部变量比使用全局变量好，因为局部变量只有线程自己能看见，不会影响其他线程，而全局变量的修改必须加锁。局部变量在一个线程中各个函数之间的传递比较麻烦。

通过 ThreadLocal 实现局部变量在函数调用中的传递。
```python
import threading
local_school = threading.local()

def process_student():
    std = local_school.student
    print('hello, %s (in %s)' % (std, threading.current_thread().name))

def process_thread(name):
    local_school.student = name
    process_student()

t1 = threading.Thread(target= process_thread, args=('Alice',), name='Thread-A')
t2 = threading.Thread(target= process_thread, args=('Bob',), name='Thread-B')
t1.start()
t2.start()
t1.join()
t2.join()
```
全局变量local_school就是一个ThreadLocal对象，每个Thread对它都可以读写student属性，但互不影响。可以把local_school看成全局变量，但每个属性如local_school.student都是线程的局部变量，可以任意读写而互不干扰，也不用管理锁的问题，ThreadLocal内部会处理。

可以理解为全局变量local_school是一个dict，不但可以用local_school.student，还可以绑定其他变量

### 进程 vs 线程
分类 | 优点 | 缺点 
---|---|---|
多线程| windows下，多线程效率高 | 稳定性差，一个线程挂掉可能导致整个进程崩溃|
多进程| 稳定性高，子进程挂掉不会影响其他进程|创建进程开销大

任务类型  | 特点 | 语言
---|---|---|
计算密集型 |大量计算，消耗cpu资源| 要求代码效率高，c语言最合适
IO密集型|cpu消耗少，主要等待IO操作完成|脚本语言

python相关模块和函数汇总：codecs模块/pickle模块
## codecs模块
python可以支持多国语言，内部使用的是unicode编码。python做编码转换的过程是：原来编码 ——>内部编码——>目的编码。codecs用来编码转换。

### 将原有编码转化为内部编码
```python
import codecs
#创建gb2312和utf-8的编码器
#lookup()在python编解码器注册表中查找编解码器信息，并返回对应信息的对象，否则出错。
look = codecs.lookup('gb2312')  #获取gb2312编解码器信息
look2 = codece.lookup('utf-8')

a = '编码器使用'
print(len(a))
print(a)

#解码为对应的编码
b = look.decode(a)  #将a解码为内部的编码：unicode
print(b[0])        #输出：b[0]为对应编码的值，b[1]为编码的长度；type编码的类型
print(b[1])
print(type(b[0]))

#将b[0]编码为gb2312类型的编码
b2 = look.encode(b[0])
print(b2[0])
print(b2[1])
print(type(b2[0]))
```
### 打开某种编码格式的文件
```python
import codecs
#open打开指定编码类型的文件
file = codecs.open('file_path','r','coding_type')
#文件打开后，读取文件时会自动转化为python内部的编码类型unicode
file_read = file.read()
```
## pickle模块
pickle是python中，压缩/保存/提取文件的模块

#### 保存文件pickle.dump()和pickle.dumps()
```python
import pickle
a_dict = {'a':1,'b':2,'c':[3,4]}
file = open('pickle_ex.pickle','wb')
pickle.dump(a_dict, file)
file.close()
```
将定义的字典a_dict保存在文件pickle_ex.pickle文件中，并生成该文件。

dumps()是将对象转化为一个字符串进行存储。

### 提取文件pickle.load()和pickle.loads()
```python
import pickle
with open('pickle_ex.pickle','rb') as file:
   a_dict = pickle.load(file)
   ##这里用with as 使用后可以自动关闭文件，不需要用户再写关闭文件操作，更简洁
```
提取文件，还原数据。

loads()表示从一个字符串中恢复对象。

说明：with ... as一般用在文件和数据库连接中，好处可以自动关闭文件通道，释放数据库连接，更加简洁。

## python3编码原理总结
* 字符编码（ASCII编码）将128个字符编码到计算机中，即大小写英文字母、数字和一些符号，ASCII编码是1个字节
* GB2312   中文超出了ACSII编的范文，中文编码GB2312
* unicode  为了统一各个国家不同的编码，产生了unicode编码，解决了乱码问题。UTF-8是unicode编码中的一个实现，UTF-8编码把一个Unicode字符根据不同的数字大小编码成1-6个字节，可以节省空间

### 默认编码
python3编译器默认的是UTF-8编码;linux操作系统默认编码为UTF-8,windows系统默认编码为GBK
```python
import sys
import locale #根据计算机用户所使用的语言、所在国家或地区，以及当地的文化传统所定义的一个软件运行时的语言环境
print(sys.getdefaultencoding())   #系统默认编码,python3编译器本身的默认编码UTF-8，python2默认为ASCII
print(locale.getdefaultlocale())  #本地默认编码，即操作系统默认的编码
```
系统默认编码与python2和python3有关
本地默认编码至于操作系统有关

### 头文件编码声明 “#coding = utf-8”
Python程序中的头文件编码声明的意思是python3编译器在读取该.py文件时，应该用什么格式解码。该声明不会改变系统默认编码和本地默认编码。
```
系统默认编码：在python3编译器读取.py文件时，若头文件没有编码声明，则使用系统默认编码对.py文件进行解码。且在调用编码encode()函数时，不知道编码方式，采用系统默认编码

本地默认编码：编写python3程序时，若使用了open()函数，不指定encoding参数，则自动使用本地默认编码。
```
### 总结
- 所有文件的编码格式都是当前编辑器决定的。当使用Python的 open( ) 函数时，是内存中的进程与磁盘的交互，而这个交互过程中的编码格式则是使用操作系统的默认编码（Linux为utf-8，windows为gbk）
- 进程在内存中的表现是“ unicode ”的编码；当python3编译器读取磁盘上的.py文件时，是默认使用“utf-8”的；当进程中出现open(), write() 这样的存储代码时，需要与磁盘进行存储交互时，则是默认使用操作系统的默认编码。
- 从网络或磁盘上读取了字节流，读到的数据类型为bytes，要将bytes变为str,需要用decode()方法解码。

### 文件操作误区
- 用python函数open()、write()存储代码时，默认使用操作系统的默认编码
- open()函数打开文件时，打开模式中带'b'表示是字节流操作，读取时，得到的是字节流类型bytes，而不是str类型，存储时，也需要转化为对应的bytes类型。通过decode()进行解码
- open()函数，打开模式中不带'b',则操作类型对应为str类型，可以指定encoding参数，设定对应的编码类型

## numpy模块函数函数
NumPy是python中的要给扩展程序库，支持高端大量的**维度数组与矩阵**运算，同时对数据运算提供大量的数学函数库。基于c语言实现，运算速度快。
注：numpy模块针对的对象事维度数组和矩阵。
### numpy.random中的shuffle和permutation
随机打乱一个数据，用于训练数据打乱样本，区别在于shuffle只对array进行打乱，且无返回值，而是对数据本身进行打乱；permutation可以对array进行打乱，也可以对传入的int类型进行打乱，且返回打乱后的array类型


## python字符串前面加r,b,u的含义
1. 字符串前面加u: 后面的字符串以unicode格式进行编码，一般用在含有中文字符串前面，防止因为源码存储格式问题，导致再次使用出现乱码

2. 字符串前面加r: 声明后面的字符串为普通字符串，一般用于将含有转义字符的字符串以普通字符串形式显示，如r"\n\t\n" ，返回"\n\t\n"

3. 字符串前面加b: python3.x中默认的str为unicode类，前面加b，将其转化为bytes类。python2.x中，str就是bytes类，故Python2.x中字符串前缀b没有什么意义，只是为了兼容python3.x的写法。

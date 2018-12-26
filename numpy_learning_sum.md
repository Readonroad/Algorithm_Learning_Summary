tensorflow模型中使用的python相关模块和函数汇总：codecs模块/pickle模块
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

## numpy模块函数函数
NumPy是python中的要给扩展程序库，支持高端大量的**维度数组与矩阵**运算，同时对数据运算提供大量的数学函数库。基于c语言实现，运算速度快。
注：numpy模块针对的对象事维度数组和矩阵。
### numpy.random中的shuffle和permutation
随机打乱一个数据，用于训练数据打乱样本，区别在于shuffle只对array进行打乱，且无返回值，而是对数据本身进行打乱；permutation可以对array进行打乱，也可以对传入的int类型进行打乱，且返回打乱后的array类型


## python字符串前面加r,b,u的含义
1. 字符串前面加u: 后面的字符串以unicode格式进行编码，一般用在含有中文字符串前面，防止因为源码存储格式问题，导致再次使用出现乱码

2. 字符串前面加r: 声明后面的字符串为普通字符串，一般用于将含有转义字符的字符串以普通字符串形式显示，如r"\n\t\n" ，返回"\n\t\n"

3. 字符串前面加b: python3.x中默认的str为unicode类，前面加b，将其转化为bytes类。python2.x中，str就是bytes类，故Python2.x中字符串前缀b没有什么意义，只是为了兼容python3.x的写法。

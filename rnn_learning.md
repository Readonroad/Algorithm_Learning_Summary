重点讲解：flags定义命令行参数/codecs模块/name_scope()和 variable_scope()/RNNCell类

## flags定义命令行参数
### 说明
通过利用tf.app.flags组件可以实现使用命令行参数，用以接受命令行传递参数

首先调用自带的DEFINE_string，DEFINE_boolean, DEFINE_integer, DEFINE_float设置不同类型的命令行参数及其默认值。当然，也可以在终端用命令行参数修改这些默认值。

定义函数说明
def DEFINE_string(flag_name, default_value, docstring): 参数分别为参数名、默认值、参数别名

### 使用
构造tensorflow命令行参数

1.导入tf
```python
import tensorflow as tf
```
2.获取使用句柄
```python
flags = tf.app.flags
```
3.定义不同类型的命令行参数
```python 
flags.DEFINE_string(flag_name, default_value, docstring)
flags.DEFINE_integer(flag_name, default_value, docstring)
...
```

解析命令行参数

1.获取使用句柄
```python
FLAGS = flags.FLAGS
```
2.获取命名行参数，可以获取不同类型的参数
```python
flag_name =  FLAGS.flag_name
```

###example
```python 
import tensorflow as tf

#构造命令行参数
flags = tf.app.flags
flags.DEFINE_string("str_test", "example", "the content of string")
flags.DEFINE_integer("batch_size", 10, "the size of bath")

#解析参数
FLAGS = flags.FLAGS
batch_size = FLAGS.batch_size
str_test = FLAGS.str_test

#使用参数 
print('batch_size = %s' %str_test)
print('batch_size = %d' %batch_size)
```
命令行中配置参数:python example.py --flag_name flag_value ...

## codecs模块
python可以支持多国语言，内部使用的是unicode编码。python做编码转换的过程是：原来编码 ——>内部编码——>目的编码。codecs用来编码转换。

###将原有编码转化为内部编码
```python
import codecs
#创建gb2312和utf-8的编码器
#lookup()在python编解码器注册表中查找编解码器信息，并返回对应信息的对象，否则出错。
look = codecs.lookup('gb2312')
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
###打开某种编码格式的文件
```python
import codecs
#open打开指定编码类型的文件
file = codecs.open('file_path','r','coding_type')
#文件打开后，读取文件时会自动转化为python内部的编码类型unicode
file_read = file.read()
```
##pickle模块
pickle是python中，压缩/保存/提取文件的模块

####保存文件
```python
import pickle
a_dict = {'a':1,'b':2,'c':[3,4]}
file = open('pickle_ex.pickle','wb')
pickle.dump(a_dict, file)
file.close()
```
将定义的字典a_dict保存在文件pickle_ex.pickle文件中，并生成该文件。

###提取文件
```python
import pickle
with open('pickle_ex.pickle','rb') as file:
   a_dict = pickle.load(file)
   ##这里用with as 使用后可以自动关闭文件，不需要用户再写关闭文件操作，更简洁
```
提取文件，还原数据。

说明：with ... as一般用在文件和数据库连接中，好处可以自动关闭文件通道，释放数据库连接，更加简洁。

## name_scope()和 variable_scope()
神经网络中结构复杂，参数较多。tf提供了通过变量名字来创建或者获取一个变量的机制。通过这个机制，在不同的函数中可以直接通过变量的名字来使用变量，而不需要将变量通过参数的形式到处传递。tf中定义变量的两种方式tf.Variable() 和 tf.get_variable()。

tf.Variable():用来创建变量，如果内部已经存在相同名字的变量，则创建一个新的变量名，原则是下划线加数字，受name_scope()和variable_scope()的影响,即会在变量的前面加上命名空间

tf.get_variable():先查看指定name的变量是否存在，如果存在，则复用，否则创建变量，不受name_scope()的影响，但是受variable_scope()的影响
```python 
#下面这两个定义是等价的
v = tf.get_variable('v', shape=[1], initializer=tf.tf.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
```
###使用方法1
get_variable()不受name_scope()的影响,Variable()受name_scope()的影响 
```python
import tensorflow as tf

with tf.name_scope('name_scope_test'):
    var1 = tf.get_variable(name = 'var1', shape =[1], dtype = tf.float32)
    var3 = tf.Variable(name = 'var2', initial_value = [2], dtype = tf.float32)
    var4 = tf.Varibale(name = 'var2', initial_value = [2], dtype =tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #或 sess.run(tf.initialize_all_variables())
    print(var1.name, sess.run(var1))
    print(var3.name, sess.run(var3))
    print(var4.name, sess.run(var4))
```
结果
```pytyhon
var1:0 [0.76692975]
name_scope_test/var2:0 [2.]
name_scope_test/var2_1:0 [2.]
```
name_space()定义了一个命名空间，Variable()定义的变量，如果名字相同，则会在该控制前中生成多个变量，用—和数字进行编号。对于get_variable()生成的变量，不受该命名空间的影响。故，不允许定义相同名字的变量
###使用方法2
get_variable()和Variable()均受variable_scope()的影响 
```python
import tensorflow as tf
with tf.variable_scope('variable_scope_test'):
	var1 = tf.get_variable(name = 'var1', shape = [1], dtype = tf.float32)
	#var5 = tf.get_variable(name = 'var1', shape = [1], dtype = tf.float32)
	var3 = tf.Variable(name = 'var2',initial_value=[2.0], dtype = tf.float32)
	var4 = tf.Variable(name = 'var2',initial_value=[2.0], dtype = tf.float32)
	
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#sess.run(tf.initialize_all_variables())
	print(var1.name, sess.run(var1))
	print(var3.name, sess.run(var3))
	print(var4.name, sess.run(var4))
```
结果：
```python
variable_scope_test/var1:0 [-0.14383781]
variable_scope_test/var2:0 [2.]
variable_scope_test/var2_1:0 [2.]
```
get_variable()和Variable()均受variable_scope()的影响, 打印结果中对变量会加上命名空间。

###使用方法3

get_variale()在一个命名空间中定义相同的变量，
```python
import tensorflow as tf
with tf.variable_scope('variable_scope_test'):
	var1 = tf.get_variable(name = 'var1', shape = [1], dtype = tf.float32)
	#var5 = tf.get_variable(name = 'var1', shape = [1], dtype = tf.float32)
	var3 = tf.Variable(name = 'var2',initial_value=[2.0], dtype = tf.float32)
	var4 = tf.Variable(name = 'var2',initial_value=[2.0], dtype = tf.float32)
##方式一：在命名空间中设置reuse=True,此时用get_variable只能获取已经存在的变量
with tf.variable_scope('variable_scope_test', reuse = True):
    var5 = tf.get_variable(name = 'var1', shape = [1], dtype = tf.float32)

##方式二：
with tf.variable_scope('variable_scope_test') as scope:
	var6 = tf.get_variable('var11', shape=[1], dtype=tf.float32)
    #下面只能用get_variable获取已经存在的变量，不能创建新的变量
	scope.reuse_variables()
	var11 = tf.get_variable(name = 'var1', shape = [1])
    var7 = tf.get_variable('var11', shape=[1], dtype=tf.float32)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#sess.run(tf.initialize_all_variables())
	print(var1.name, sess.run(var1))
	print(var3.name, sess.run(var3))
	print(var4.name, sess.run(var4))
```
结果：
```python
#方式一
variable_scope_test/var1:0 [-1.4252254]
variable_scope_test/var2:0 [2.]
variable_scope_test/var2_1:0 [2.]
variable_scope_test/var1:0 [-1.4252254]
#方式二
variable_scope_test/var1:0 [0.96736276]
variable_scope_test/var2:0 [2.]
variable_scope_test/var2_1:0 [2.]
variable_scope_test/var11:0 [-0.56519735]
variable_scope_test/var1:0 [0.96736276]
```
在命名空间内设置了reuse = True,tf.get_variable只能获取**已经存在的变量**而不能**创建变量**。

## RNNCell类

RNNCell只是一个抽象类，我们用的时候都是用的它的两个子类BasicRNNCell和BasicLSTMCell。没调用一次RNNCell的call方法，相当于在时间删个“推进一步”。

### 单步RNN:BasicRNNCell和BasicLSTMCell

使用方法：
```python
#BasicRNNCell
import tensorflow as tf
import numpy as np

cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 128) #隐藏层的尺寸为128
print(cell.state_size)  

inputs = tf.placeholder(np.float32, shape=(32,100))  #输入的尺寸为32*100
h0 = cell.zero_state(32,np.float32) #zero-state()得到一个全0的初始状态32*128
output,h1 = cell.call(inputs, h0)  #(output,next_state)=call(input, state)

print(h1.shape)    #调用call后隐藏层的输出尺寸为32*128,32为batch_size

#BasicLSTMCell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = lstm_cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell.call(inputs, h0)

#lstm的输出中有两个隐藏状态h和c,对应得到的隐藏层h1是一个tuple,每个元素的尺寸为（batch_size,state_size)
print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)
```
### 多步执行：tf.nn.dynamic_rnn
将输入inputs的数据格式定义为（batch_size, time_steps,input_size),其中time_steps为序列本身的长度，则用tf.nn.dynamic_rnn()可以多次调用time_steps次，输出outputs为time_steps步中所有的输入，state为最后一步的隐状态。

使用方法：
```python
#BasicRNNCell
import tensorflow as tf
import numpy as np

cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 128) #隐藏层的尺寸为128
print(cell.state_size)  

inputs = tf.placeholder(np.float32, shape=(32,200,100))  #输入的尺寸为32*100
h0 = cell.zero_state(32,np.float32) #zero-state()得到一个全0的初始状态32*128
outputs, state = tf.nn.dynamic_rnn(cell, inputs, h0) 
print(h1.shape)    #调用call后隐藏层的输出尺寸为32*128,32为batch_size

#BasicLSTMCell
#lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
#h0 = lstm_cell.zero_state(32, np.float32) #通过zero_state得到一个全0的初始状态
```
### 堆叠RNNCell:MultiRNNCell

多层的RNN,可以提高RNN的性能，使用tf.nn.rnn_cell.MultiRNNCell函数对RNNCell进行堆叠。

使用方法：
```python
import tensorflow as tf
import numpy as np

#定义一个基本的BasicRNNCell
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units = 128)
#调用MultiRNNCell创建多层RNN
cell =  tf.nn.run_cell.MultiRNNCell([get_a_cell() for _ in range(3)])
print(cell.state_size)    #三个隐层状态，每个隐层状态大小为128

inputs = tf.placeholder(np.float32, shape=(32,100))  #输入的尺寸为32*100
h0 = cell.zero_state(32,np.float32) #zero-state()得到一个全0的初始状态32*128
outputs, state = cell.call(inputs, h0) 
print(h1)    #隐藏层输出是一个tuple，含有3个32*128的向量
```
多层的RNN如果想要进行多步执行，要需要使用tf.nn.dynamic_rnn().

### 注意事项
1. 调用call()或dynamic_rnn()函数后得到的output并不是最终的输出，在tf的实现中，output和state是相同的，如果要得到真正的输出，需要额外对输出定义新的变换，才能得到最后的输出;

2. 版本不同引起的错误，注意不同的版本，使用的函数或者参数可能不同。

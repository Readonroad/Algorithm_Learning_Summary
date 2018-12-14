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
## pickle模块
pickle是python中，压缩/保存/提取文件的模块

#### 保存文件
```python
import pickle
a_dict = {'a':1,'b':2,'c':[3,4]}
file = open('pickle_ex.pickle','wb')
pickle.dump(a_dict, file)
file.close()
```
将定义的字典a_dict保存在文件pickle_ex.pickle文件中，并生成该文件。

### 提取文件
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

## one_hot()
```python
import tensorflow as tf
tf.one_hot(indices, depth, on_value, off_value, axis)
```
indices表示一个列表，指定张量中独热向量的独热位置，独热位置处填入独热值on_value,其他位置填入off_value. axis指定第几阶独热向量，-1指定张量的最后一维为独热向量，1表示行向量。

## tf.nn.embedding_lookup
```python
tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
```
embedding就是将word映射为向量，使用embedding技术使词与词产生的向量之间存在某种联系，即意思相近的词产生的向量在空间上更加接近。
```python
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

embedding = tf.Variable(np.identity(6, dtype=np.int32))
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

sess.run(tf.global_variables_initializer())
print sess.run(embedding)
print sess.run(input_embedding, feed_dict={input_ids: [4, 0, 2, 4, 5, 1, 3, 0]})
```
结果：
```python
[[1 0 0 0 0 0]
 [0 1 0 0 0 0]
 [0 0 1 0 0 0]
 [0 0 0 1 0 0]
 [0 0 0 0 1 0]
 [0 0 0 0 0 1]]
 #input_ids中的值表示取embedding中的第ids行
[[0 0 0 0 1 0]
 [1 0 0 0 0 0]
 [0 0 1 0 0 0]
 [0 0 0 0 1 0]
 [0 0 0 0 0 1]
 [0 1 0 0 0 0]
 [0 0 0 1 0 0]
 [1 0 0 0 0 0]]
```
## tf.nn.softmax_cross_entropy_with_logits()
神经网络，交叉熵作为损失函数

1. 对输出做softmax变换，转化为概率值
```python
logits = tf.matmul(x, w) + b
 y = tf.nn.softmax(logits)
```
2. 求交叉熵
```python
 #交叉熵函数，计算结果为向量  
cross_entropy = tf.nn.softmax_cross_entropy_logits(logits, y_true)
```  
3. 求和
```python
loss = tf.reduce_mean(cross_entropy)
#直接计算,利用交叉熵公式计算，输入样本的交叉熵求均值
loss = -tf.reduce_mean(y*tf.log(y))
```
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

## tf中获取tensor和模型参数 
1. 查看tf中tensor,variable,constant的值
```python
import tensorflow as tf

with tf.Session() as sess:
    ts_res = sess.run(ts)  #查看tensor的值，返回一个narray格式的对象
    print(ts_res)

    input = {}   #输入feed的数据
    ts_res2 =  sess.run(ts,feed_dict = input)
    print(ts_res)
```

2. 如何获取模型参数

模型训练完成后，通常将模型参数保存在/checkpoint/xxx.model文件。

通过tf.train.Saver()恢复模型参数

saver = tf.train.Saver()

通过saver的restore()方法可以从本地的模型文件中恢复模型参数
```python
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(...)
b = tf.Variable(...)

y_ = tf.add(b, tf.matmul(x, w))

# create the saver

saver = tf.train.Saver()

# creat the session you used in the training processing
# launch the model

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/your/path/model.ckpt")
  print("Model restored.")
  # Do some work with the model, such as do a prediction
  pred = sess.run(y_, feed_dict={batch_x})
  ...
```

tf.traibable_variables()方法将返回模型中所有**可训练的参数**。

## tf.gradients计算导数和gradient clipping解决梯度爆炸/消失问题
训练神经网络时，经常遇到梯度爆炸和梯度消失的问题，梯度爆炸即某个梯度的值极大，导致梯度越来越大，梯度消失即某个梯度趋于0，导致梯度越来越小。可以将梯度控制在一定范围内，从而解决梯度爆炸和梯度消失的问题。

方法：

1. 计算trainable_variables集合中所有可训练的参数的梯度
```python
params =  tf.trainable_variables()
```
2. 将梯度应用到变量上进行梯度下降
```python
gradients = tf.gradients(loss, params)  #对所有可训练的变量求梯度
```
3. gradient clipping对梯度进项裁剪
```python
#裁剪函数1
clipped_t = tf.clip_by_value(t, clip_value_min,clip_value_max, name = None)
if   t > clip_value_max
    clipped_t = clip_value_max
else if  t < clip_value_min
    clipped_t = clip_value_min
else 
    clipped_t = t

#裁剪函数2,norm表示l2范数
clipped_t = tf.clip_by_norm(t, clip_norm, axes = None, name = None) 
if   clip_norm >= l2_norm
           clipped_t = t
else
    clipped_t = t*clip_norm/l2norm(t)

#裁剪函数4,norm表示l2范数
clipped_t = tf.clip_by_average_norm(t_list, clip_norm, name = None)
if   clip_norm >= l2_norm
           clipped_t = t
else
    t = t*clip_norm/l2norm_avg(t)

 #裁剪函数4,norm表示l2范数
clipped_t, global_norm = tf.clip_by_global_norm(t_list, clip_norm, use_norm =  None, name = None)
#返回结果： t_list[i] * clip_norm / max(global_norm, clip_norm) 
global_norm = sqrt(sum([l2norm(t)**2 for t in t_list])) 
```
## 梯度下降优化器
梯度下降法优化：梯度计算、梯度更新，compute_gradients()和apply_gradients()

1. 使用minimize()
```python
#定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01) 
#最小化
train_op = optimizer.minimize(loss)
```
2. compute_gradients()+apply_gradients() = minimize()
```python
#定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01) 
#优化
gradient_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(gradient_vars)
```
3. 基础梯度计算
```python
train_vars = tf.trainable_variables()          #获取所有可训练变量
gradient_vars = tf.gradients(loss, train_vars) #计算变量梯度
grads, _ = tf.clip_by_global_norm(gradient_vars, clip_norm) #梯度裁剪

#优化梯度
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
#更新变量
train_op =  optimizer.apply_gradients(zip(grads, train_vars))
```
4. 优化器梯度操作
```python
#定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
#计算梯度,compute_gradients()计算梯度，返回的是一个列表，其中元素类型为（梯度，变量）的tuple,但是该函数可能返回（None,变量）,即部门变量没有对应的梯度，下一步NoneType会导致错误。故，计算完梯度，需要过滤
gradient_all = optimizer.compute_gradients(loss)
#剔除没有梯度的变量,得到可以计算梯度的变量
gradient_vars =  [v for (g,v) in gradient_all if g is not None]
#计算变量的梯度
gradients  = optimizer.compute_gradients(loss, gradient_vars)
#应用梯度
train_Op = optimizer.apply_gradients(zip(gradients, gradient_vars))
```
## tf保存和加载文件

### tf保存文件
tensorflow模型中保存的文件有四种：
.meta / .data  /.index / checkpoint file

1 .meta扩展名文件：保存了完整的tensorflow图，即变量，操作和集合

二进制文件，保存了模型中所有的权重，偏值，梯度和其他变量的值
2 .data扩展名文件和.index扩展名文件

3 checkpoint文件：记录了最近一次保存的checkpoint文件，且必须有该文件,该文件保存了一个目录下所有的模型文件列表,该文件由tf.train.Saver自动生成且自动维护的。

```python
#保存文件
saver = tf.train.Saver()
sess = tf.Session()
saver.save(sess, 'my_train_model')
"""
保存文件类型如下：
# my_train_model.data-00000-of-00001
# my_train_model.index
# my_train_model.meta
# checkpoint
saver.save(sess, 'my_train_model',global_step = 1000)
global_step指定保存该迭代次数的结果
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
指定保存模型的个数和保存模型的时间间隔
"""
```
### tf加载文件
当需要用别人预训练好的模型进行fine-tuning时，需要加载模型

1.创造模型

可以使用python写好和原来模型一样的每一层代码构造网络，或者直接通过保存的数据文件再创造网络。
```python
#导入.meta文件,创造网络，但是不能得到训练好的参数
saver = tf.train.import_meta_graph('xxx.meta')
```
2.加载参数 

创造好网络后，还需要恢复网络中已经训练好的参数
```python
 with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('xxx.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./')
```

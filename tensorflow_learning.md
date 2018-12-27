tensorflow基本知识
## tensorflow基本简介
tensorflow由google开发的用于神经网络的开源软件库，可以先构造计算结构图，再在会话中执行构造的机构图，其中提供了大量都构造神经网络的软件库，减低了深度学习的开发成本和开发难度。

* tensorflow使用图(graph)来表示计算
* 在会话(session)中执行图
* 使用张量(tensors)表示数据
* 使用变量(variables)维护状态，定义模型参数
* 使用供给(feed)和取回(fetch)将数据传入或传出任何操作

tensorflow采用数据流图计算，图中节点表示operation,线表示数据，即tensors。一个operation可以获得0个或者多个张量执行计算,产生0个或者多个张量。训练模型时，tensor会不断从数据流图中的一个节点flow到另一个节点，即tensorflow的由来。

### tensorlow --- variable
tf中定义变量的两种方式tf.Variable() 和 tf.get_variable()。
```python 
#下面这两个定义是等价的
v = tf.get_variable('v', shape=[1], initializer=tf.tf.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
# tf.constant()
tf.constant(value, dtype=None, shape = None, name ='const')
#创建一个常量张量，值类型为dtype的value
```
tensorflow中定义变量后，只有在session中激活后，才能得到对应的值。
#### name_scope()和 variable_scope()
神经网络中结构复杂，参数较多。tf提供了通过变量名字来创建或者获取一个变量的机制。通过这个机制，在不同的函数中可以直接通过变量的名字来使用变量，而不需要将变量通过参数的形式到处传递。

tf.Variable():用来创建变量，如果内部已经存在相同名字的变量，则创建一个新的变量名，原则是下划线加数字，受name_scope()和variable_scope()的影响,即会在变量的前面加上命名空间

tf.get_variable():先查看指定name的变量是否存在，如果存在，则复用，否则创建变量，不受name_scope()的影响，但是受variable_scope()的影响

##### 使用方法1
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
##### 使用方法2
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

##### 使用方法3
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
### flags定义命令行参数
通过利用tf.app.flags组件可以实现使用命令行参数，用以接受命令行传递参数

首先调用自带的DEFINE_string，DEFINE_boolean, DEFINE_integer, DEFINE_float设置不同类型的命令行参数及其默认值。当然，也可以在终端用命令行参数修改这些默认值。

定义函数说明
def DEFINE_string(flag_name, default_value, docstring): 参数分别为参数名、默认值、参数别名

#### 使用说明
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

#### example
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

### 会话：tf.Session() 和 tf.InteractiveSession()
tensorflow构造计算图后，需要在会话session中执行图，在session中反复执行图中的训练operation。
在我们使用tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作(operation),如果我们使用tf.Session()来构建会话我们需要在会话构建之前定义好全部的操作(operation)然后再构建会话。
#### tf.Session()
Session对象使用完后需要关闭以释放资源，可以显示调用close()或者用with代码块自动关闭。在启动Session()之前需要先构造整个计算图。
```python
sess = tf.Session()  #创建一个Session对象，若无参数，将启动默认图
sess.run(op)   #执行图中的operation
sess.close()   #显示关闭会话
#利用with代码块自动关闭会话
with tf.Session() as sess:
    with tf.device("/gpu:1"):   #指定第二个gpu参与计算
        sess.run(op)
#设备用字符串进行标识
#"/cpu:0" :机器的cpu
#"/gpu:0" :机器的第一个gpu
#"/gpu:1" :机器的第二个gpu
```
在实现上，tensorflow将图形定义转化成分布式执行的操作，以充分利用可用的计算资源（CPU、GPU)，一般不需要显示指定使用CPU还是GPU，tensorflow能自动检测，如果检测到GPU，tensorflow会尽可能地利用找到的第一个GPU来执行操作.若机器上有超过一个可用的GPU，除第一个外的其他GPU默认是不参与计算的。可以使用with...Device语句指派特定的CPU或GPU执行操作。

#### tf.InteractiveSession() 交互式Session
使用交互式的方式，可以避免使用一个变量来持有会话。可以使用Tensor.eval()和Operation.run()的方法代替Session.run()。在运行图的时候，插入一些计算图，更加灵活构造代码。

##### .eval()和.run()
eval()实质上是tf.Tensor类的Session.run()的另一种写法，区别如下：

1. eval()：将字符串string对象转化为有效的表达式参与求值运算返回计算结果；
2. 它只能用于tf.Tensor类对象，即必须是有输出的Operation，对于没有输出的Operation，可以使用run()或Session.run()。
3. Session.run([tensor1,...,tensorn])可以一次获取多个tensor的值，而tensor.eval()每次只能获取一个tensor的值。

**每次执行eval（）和run()的时候，都会执行整个计算图。**

### 传入feed()和取回fetch()
fetch操作指Tensorflow的session可以一次run多个operation。

feed操作是指首先建立占位符，然后把占位符放入operation中，在run(operation)时，再把operation的值传进去，以达到使用时再传参数的目的。
#### fetch()
语法：将多个operation放入数组中传给run()方法。
```python
import tensorflow as tf

#Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

#定义两个op
add = tf.add(input2, input3)
mu1 = tf.multiply(input1, add)

with tf.Session() as sess:
    #一次操作两个op, 按顺序返回结果,即为fetch
    result = sess.run([mu1, add])
    print(result)
```
#### feed()
语法：首先创建placeholder，然后再在run的时候把值以字典的方式传入run
```python
import tensorflow as tf
#定义数据
x y
#Fetch
input1 = tf.placeholder(tf.float32,shape=[3,3])
input2 = tf.placeholder(tf.float32,shape=[3,3])

#定义两个op
add = tf.add(input1, input2)
mu1 = tf.multiply(input1, add)

with tf.Session() as sess:
    #一次操作两个op, 按顺序返回结果,即为fetch
    result = sess.run([mu1, add],feed_dict={input1:x,input2:y})
    print(result)
```

### one_hot()
```python
import tensorflow as tf
tf.one_hot(indices, depth, on_value, off_value, axis)
```
indices表示一个列表，指定张量中独热向量的独热位置，独热位置处填入独热值on_value,其他位置填入off_value. axis指定第几阶独热向量，-1指定张量的最后一维为独热向量，1表示行向量。

### tf.gradients计算导数和gradient clipping解决梯度爆炸/消失问题
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

#裁剪函数3,norm表示l2范数
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
### 梯度下降优化器
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


### tf保存和加载文件

#### tf保存文件
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
#### tf加载文件
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
### tensorflow中构造神经网络相关函数
#### 交叉熵损失：tf.nn.softmax_cross_entropy_with_logits()
神经网络，交叉熵作为损失函数

1. 对输出做softmax变换，转化为概率值
```python
logits = tf.matmul(x, w) + b
 y = tf.nn.softmax(logits)
```
2. 求交叉熵
```python
 #交叉熵函数，计算结果为向量  
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_true)
```  
3. 求和
```python
loss = tf.reduce_mean(cross_entropy)
#直接计算,利用交叉熵公式计算，输入样本的交叉熵求均值
loss = -tf.reduce_mean(y*tf.log(y))
```
#### dropout层
dropout层大量用于全连接网路，一般设置为0.5或0.3，卷积网络隐藏层中由于卷积自身的稀疏或稀疏化的relu函数的大量使用，Dropout策略在卷积网路隐藏层中使用较少。dropout是一个超参，需要根据具体的网络和具体应用领域进行尝试。
```python
keep_prob = tf.placeholder(tf.float32)   #保留神经元的比例
Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
```
训练时，针对dropout后的参数调整，而测试的时候，会让所有神经元权重乘以概率p对测试样本预测。
#### cnn相关函数

##### 卷积:tf.nn.conv2d()
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu = None, name = None)

前四个参数比较重要：

input[batch,长,宽,通道数]

filter[长,宽,通道数,输出通道数]

strides[1,长步长,宽步长,1]

padding：卷积核在边缘处的处理方法，如‘SAME'/'VALID'

use_cudnn_on_gpu 是否使用cudnn加速，默认为true
##### 激活函数
tf.nn.relu() / tf.nn.sigmoid() / tf.nn.tanh()
##### 池化
max_pool / avg_pool 

tf.nn.max_pool(input, [1,pool_size,pool_size,1], [1,rstride,cstride,1] , padding) 

pool_size是pool的大小，rstride和cstride分别是行列上的步长，padding可取SAME或VALID
#### RNNCell类
RNNCell只是一个抽象类，我们用的时候都是用的它的两个子类BasicRNNCell和BasicLSTMCell。每调用一次RNNCell的call方法，相当于在时间上“推进一步”。

##### 单步RNN:BasicRNNCell和BasicLSTMCell

使用方法：
```python
#BasicRNNCell
import tensorflow as tf
import numpy as np

cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 128) #隐藏层的尺寸为128
print(cell.state_size)  

inputs = tf.placeholder(np.float32, shape=(32,100))  #输入的尺寸为32*100
h0 = cell.zero_state(32,np.float32) #zero_state()得到一个全0的初始状态32*128
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
##### 多步执行：tf.nn.dynamic_rnn
将输入inputs的数据格式定义为（batch_size, time_steps,input_size),其中time_steps为序列本身的长度，则用tf.nn.dynamic_rnn()可以调用time_steps次，输出outputs为time_steps步中所有的输入，state为最后一步的隐状态。

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
##### 堆叠RNNCell:MultiRNNCell

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

##### 注意事项
1. 调用call()或dynamic_rnn()函数后得到的output并不是最终的输出，在tf的实现中，output和state是相同的，如果要得到真正的输出，需要额外对输出定义新的变换，才能得到最后的输出;

2. 版本不同引起的错误，注意不同的版本，使用的函数或者参数可能不同。

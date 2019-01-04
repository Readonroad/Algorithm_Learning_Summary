神经网络可视化 Tensorboard 和 tf.summary

标量：0阶张量，表示值；向量是一阶张量，表示方向；张量表示整个空间。
## Summary & TensorBoard
Summary是对网络中Tensor取值进行监测的一种operation,这些操作在图中是“外围"操作，不影响数据流本身。注：Summary本身是一种operation。

TensorBoard可以把复杂的神经网络训练过程可视化，更好理解、调试、优化程序。TensorBoard可视化过程需要使用Summary中的方法。

## Summary Operations
### summary operations中记录信息的节点
1） tf.summary.scalar 

监测标量，如learning_rate，loss等

2）tf.summary.histogram

查看activations,gradients,weights,bias等张量的直方图

3） tf.summary.distribution

记录activations,gradients,weights,bias等张量的分布

4）tf.summary.image

记录图像数据
### summary operations执行
Summary Operations并不会真的执行，需要执行run(op) 或者被其他需要run的operation所依赖时，该op才会执行。程序中如果有太多的summary节点，手动启动每个节点过于繁琐，因此可以使用tf.summary.merge_all将所有的节点合并为一个节点，只要运行这个节点，就能产生所有设置的summary data.
### summary data 保存
使用tf.summary.FileWriter将summary data写入磁盘。tf.summary.FileWrite中包含了参数logdir，所有的事件都会写到它所指的目录下。
## TensorBoard
### TensorBoard的数据形式
TensorBoard可以显示和展示以下数据

* 标量Scalars
* 图片Images
* 音频Audio
* 计算图Graph
* 数据分布Distribution
* 直方图Histograms
* 嵌入向量Embeddings

### TensorBoard数据可视化过程
1）首先建立一个graph,标记需要获取数据的信息

2）确定图中哪些节点放置summary operations以记录信息，如tf.summary.scalar,tf.summary.histogram,tf.summary.distribution,tf.summary.iamge等

3）执行summary operations

4）使用tf.summary.FileWriter写入本地磁盘中

5）运行程序，并在命令行中运行tensorboard的指令，然后在web页面查看可视化结果，命令：tensorboard --logdir [log目录],web页面地址：http://localhost:6006

## simple example
mnist_summary.py
```python
#coding=gbk
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#初始化会话
sess = tf.InteractiveSession()

#加载数据
fashion_mnist =  input_data.read_data_sets("data/MNIST_data/", one_hot = True)  
train_images = fashion_mnist.train.images #训练数据，得到的是np array 数据类型  55000*784
train_labels = fashion_mnist.train.labels  #训练数据的标签 55000*10

test_image = fashion_mnist.test.images        #测试数据，10000*784
test_labels = fashion_mnist.test.labels

#定义变量
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1) #截断高斯分布参数 
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)
	
#定义二维卷积层
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

#定义max-pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1], padding ='SAME')

#graph = tf.Graph()    #定义计算图
#with graph.as_default():   #返回计算图的上下文管理器
#定义占位符
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None,784])
	y_ = tf.placeholder(tf.float32, [None,10])
	x_image = tf.reshape(x, [-1, 28,28,1]) 
	tf.summary.image('input',x_image,10)

#定义第一个卷积层
shape1 = [5,5,1,16]  #输入尺寸为5*5，深度为1,大小为32的卷积核
with tf.name_scope('layer1_weight'):
	W_conv1 = weight_variable(shape1)
	tf.summary.histogram('layer1/weights',W_conv1)
with tf.name_scope('layer1_bias'):
	b_conv1 = bias_variable([16])
	tf.summary.histogram('layer1/bias',b_conv1)
with tf.name_scope('layer1_relu'):
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1) #relu激活函数
#pooling
with tf.name_scope('layer1_pool'):
	h_pool1 = max_pool_2x2(h_conv1) #卷积后的pooling

#定义第二个卷积层
shape2 = [5,5,16,32]  #输入尺寸为5*5，深度为32,大小为64的卷积核
with tf.name_scope('layer2_weight'):
	W_conv2 = weight_variable(shape2)
	tf.summary.histogram('layer2/weights',W_conv2)
with tf.name_scope('layer2_bias'):
	b_conv2 = bias_variable([32])
	tf.summary.histogram('layer2/bias',b_conv2)
with tf.name_scope('layer2_relu'):
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+ b_conv2) #relu激活函数
#pooling
with tf.name_scope('layer2_pool'):
	h_pool2 = max_pool_2x2(h_conv2) #卷积后的pooling

#定义全连接层
shape_fc1 = [7*7*32, 1024] #全连接层大小为上一层输入的大小
W_fc1 = weight_variable(shape_fc1)
tf.summary.histogram('layer3/weights',W_fc1)
b_fc1 = bias_variable([1024])
tf.summary.histogram('layer3/bias',b_fc1)
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*32]) #卷积层输出变量列向量
h_fc1 =  tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  #全连接层输出

#加一层dropput层，减少参数个数，防止过拟合
keep_pro =  tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_pro)

#dropout层后，连接softmax层输出，得到最后的输出概率
shape_fc2 = [1024,10]     #输入结果对应类别个数
W_fc2 = weight_variable(shape_fc2)
tf.summary.histogram('layer4/weights',W_fc2)
b_fc2 = bias_variable([10])
tf.summary.histogram('layer4/bias',b_fc2)
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#定义损失函数
loss = -tf.reduce_mean(y_ *tf.log(y_conv))
tf.summary.scalar('loss',loss)
#最优化
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
train_op = optimizer.minimize(loss)

#定义准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))   #argmax(input,axis = None)  指定某个维度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   #tf.cast(x,dtype) 对输入x进行类型转换
tf.summary.scalar('train_arruary',accuracy)

merged = tf.summary.merge_all()      #合并所有的summary data
write = tf.summary.FileWriter('log/',graph = tf.get_default_graph())   #tf.get_default_graph()得到默认的计算图
tf.global_variables_initializer().run()    #参数初始化
for i in range(2000):      #range()从0开始，不包括结束数
	batch = fashion_mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy =  accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_pro:1.0})   #测评时，keep_prob设为1，即不dropout，监控模型的性能
		print("step %d, training accuracy %g" %(i, train_accuracy))
		rs = merged.eval(feed_dict={x:batch[0], y_:batch[1], keep_pro:0.5})   #tensor需要用eval，或者sess.run(mereged,...)
		write.add_summary(rs,i)    #保存数据
	train_op.run(feed_dict={x:batch[0], y_:batch[1], keep_pro:0.5})     #训练的时候，keep_prob设为0.5，防止过拟合

#测试样本性能
print("test accuracy %g" %accuracy.eval(feed_dict={x:test_image,y_:test_labels,keep_pro:1.0}))
```

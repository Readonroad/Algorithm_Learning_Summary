import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

fashion_mnist =  input_data.read_data_sets("MNIST_data/", one_hot = True)  #读取mnist数据，本地下载的mnist不需要解压

train_images = fashion_mnist.train.images     #训练数据，得到的是np array 数据类型  55000*784
train_labels = fashion_mnist.train.labels     #训练数据的标签 55000*10
 
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

#定义占位符
x = tf.placeholder(tf.float32, [None,784])
y_ = tf.placeholder(tf.float32, [None,10])
x_image = tf.reshape(x, [-1, 28,28,1])  #将输入reshape成二维图像28*28，-1表示样本个数不固定

#定义第一个卷积层
shape1 = [5,5,1,32]  #输入尺寸为5*5，深度为1,大小为32的卷积核
W_conv1 = weight_variable(shape1)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1) #relu激活函数
#pooling
h_pool1 = max_pool_2x2(h_conv1) #卷积后的pooling

#定义第二个卷积层
shape2 = [5,5,32,64]  #输入尺寸为5*5，深度为32,大小为64的卷积核
W_conv2 = weight_variable(shape2)
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+ b_conv2) #relu激活函数
#pooling
h_pool2 = max_pool_2x2(h_conv2) #卷积后的pooling

#定义全连接层
shape_fc1 = [7*7*64, 1024] #全连接层大小为上一层输入的大小
W_fc1 = weight_variable(shape_fc1)
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64]) #卷积层输出变量列向量
h_fc1 =  tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  #全连接层输出

#加一层dropput层，减少参数个数，防止过拟合
keep_pro =  tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_pro)

#dropout层后，连接softmax层输出，得到最后的输出概率
shape_fc2 = [1024,10]     #输入结果对应类别个数
W_fc2 = weight_variable(shape_fc2)
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#定义损失函数
loss = -tf.reduce_mean(y_ *tf.log(y_conv))
#最优化
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train_op = optimizer.minimize(loss)

#定义准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   #tf.cast(x,dtype) 对输入x进行类型转换
	
tf.global_variables_initializer().run()    #参数初始化
for i in range(20000):      #range()从0开始，不包括结束数
	batch = fashion_mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy =  accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_pro:1.0})   #测评时，keep_prob设为1，即不dropout，监控模型的性能
		print("step %d, training accuracy %g" %(i, train_accuracy))
	train_op.run(feed_dict={x:batch[0], y_:batch[1], keep_pro:0.3})     #训练的时候，keep_prob设为0.5，防止过拟合
	
#测试样本性能
print("test accuracy %g" %accuracy.eval(feed_dict={x:test_image,y_:test_labels,keep_pro:1.0}))

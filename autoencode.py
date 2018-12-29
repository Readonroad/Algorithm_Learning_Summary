#coding=gbk
##自编码实现：对784输入编码为256，再编码为128，后解码，激活函数为sigmoid
'''
注：自编码是对输入进行编码，再解码的过程，输入与输出的均方误差最小化，一种非监督的方法
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

#训练参数
learning_rate = 0.001
train_num = 15000
display_step = 20
batch_size = 50

#模型参数
num_input = 28*28
num_hidden1 = 256
num_hidden2 = 128
num_labels = 10

#加载数据
fashion_mnist =  input_data.read_data_sets("data/MNIST_data/", one_hot = True)  
train_images = fashion_mnist.train.images #训练数据，得到的是np array 数据类型  55000*784
train_labels = fashion_mnist.train.labels  #训练数据的标签 55000*10

test_image = fashion_mnist.test.images        #测试数据，10000*784
test_labels = fashion_mnist.test.labels

#定义编码解码的参数
weights={
		'en1_w':tf.Variable(tf.truncated_normal(shape=[num_input,num_hidden1])),
		'en2_w':tf.Variable(tf.truncated_normal(shape=[num_hidden1,num_hidden2])),
		'de1_w':tf.Variable(tf.truncated_normal(shape=[num_hidden2,num_hidden1])),
		'de2_w':tf.Variable(tf.truncated_normal(shape=[num_hidden1,num_input]))}
biases={
		'en1_b':tf.Variable(tf.truncated_normal(shape=[num_hidden1])),
		'en2_b':tf.Variable(tf.truncated_normal(shape=[num_hidden2])),
		'de1_b':tf.Variable(tf.truncated_normal(shape=[num_hidden1])),
		'de2_b':tf.Variable(tf.truncated_normal(shape=[num_input]))}

def encoder(x):
	en_layer1_output = tf.nn.sigmoid(tf.matmul(x, weights['en1_w'])+biases['en1_b'])
	en_output = tf.nn.sigmoid(tf.matmul(en_layer1_output, weights['en2_w'])+biases['en2_b'])
	return en_output

def decoder(x):
	de_layer1_output = tf.nn.sigmoid(tf.matmul(x, weights['de1_w'])+biases['de1_b'])
	de_output = tf.nn.sigmoid(tf.matmul(de_layer1_output, weights['de2_w'])+biases['de2_b'])
	return de_output

#定义占位符
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, num_input])   #rnn输入数据num_seqs*num_step(序列长度)*num_input(输入长度)

en_output =  encoder(x)
de_output = decoder(en_output)
x_pre = de_output
#定义损失函数:均方误差
loss = tf.reduce_mean(tf.pow(x_pre-x, 2))
#最优化
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()   #参数初始化
with tf.Session() as sess:
	init.run()
	step = 1
	for step in range(1,train_num+1):
		batch = fashion_mnist.train.next_batch(batch_size)
		
		train_op.run(feed_dict={x:batch[0]}) 
		if step % display_step == 0 or step == 1:
			train_loss = sess.run(loss, feed_dict={x:batch[0]}) 
			#测试样本性能
			print("step %d, minibatch loss : %g" %(step, train_loss))
		step+=1
	print("optimization finised!")
	
	#对测试样本进行从重构
	n = 4
	container_origs = np.empty((28*n,28*n))   #定义放置n个图像的容器
	container_recons = np.empty((28*n,28*n))
	
	for i in range(n):
		test_batch = fashion_mnist.test.next_batch(n)
		test_recon = sess.run(de_output, feed_dict={x:test_batch[0]})   #重构后的输出
		
		#画出对比图
		for j in range(n):
			container_origs[i*28:(i+1)*28,j*28:(j+1)*28] = test_batch[0][j].reshape([28,28])
			container_recons[i*28:(i+1)*28,j*28:(j+1)*28] = test_recon[j].reshape([28,28])
	#画图
	plt.figure(figsize=(n,n))
	plt.subplot(1,2,1)
	plt.imshow(container_origs, origin='upper',cmap='gray')
	plt.title('origin images')
	#plt.show()

	plt.subplot(1,2,2)
	plt.imshow(container_recons, origin='upper',cmap='gray')
	plt.title('reconstructed images')
	plt.show()
	
	#print(test_batch[0][1])

#coding=gbk
##通过RNN模型对MNIST数据进行分类
'''
遇到的坑：
1.outputs,state = tf.nn.dynamic_rnn(cell, inputs, state0)
正确写法：outputs,state = tf.nn.dynamic_rnn(cell, inputs, initial_state = state0)
initial_state是可选参数，需要加上变量名，否则会按顺序识别为其他变量
2.tf.nn.dynamic_rnn()的输出，Outputs是包含num_step次的输出，而state是最后一次的状态，都不是最后的数据
需要对最后一次的结果进行变换，得到最终的结果。另外，对于lstm，state有两个状态，state[0]为状态c,state[1]为状态h。
3.测试的时候，需要从测试样本中选取batch_size个样本进行测试
由于rnn_model中隐藏状态state0初始化时大小为batch_size。
疑问：
是否可以针对所有样本进行测试？？？
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#训练参数
learning_rate = 0.001
train_num = 15000
display_step = 20
batch_size = 50

#模型参数
num_step = 28   #次数
num_input = 28
num_hidden = 128
num_labels = 10

#加载数据
fashion_mnist =  input_data.read_data_sets("data/MNIST_data/", one_hot = True)  
train_images = fashion_mnist.train.images #训练数据，得到的是np array 数据类型  55000*784
train_labels = fashion_mnist.train.labels  #训练数据的标签 55000*10

test_image = fashion_mnist.test.images        #测试数据，10000*784
test_labels = fashion_mnist.test.labels

#定义占位符
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, num_step, num_input])   #rnn输入数据num_seqs*num_step(序列长度)*num_input(输入长度)
	y_ = tf.placeholder(tf.float32, [None, num_labels])
#定义变量,对于cell输出的结果要进一步进行变换得到最终的输出
weights = {'out':tf.Variable(tf.truncated_normal(shape=[num_hidden,num_labels]))}
biases = {'out':tf.Variable(tf.truncated_normal(shape=[num_labels]))}

#定义RNN模型
def rnn_model(inputs, weights, biases):
	#输入是num_seqs * num_step,num_input的格式
	#cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)   #基本的RNN处理单元
	cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)
	state0 = cell.zero_state(batch_size, tf.float32) #全0的初始状态
	outputs,state = tf.nn.dynamic_rnn(cell, inputs, initial_state = state0) #outputs是num_step次的所有输出结果；state为最后一次的隐状态
	#LSTM输出的状态有2个，state[0]是cell state,state[1]是隐藏层最终的状态
	state = tf.matmul(state[1],weights['out'])+ biases['out']
	return state

#定义损失函数
logits = rnn_model(x, weights, biases)
y = tf.nn.softmax(logits)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_)) #注意函数内部带上变量名
tf.summary.scalar('loss',loss)
	
#最优化
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

#定义准确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))   #argmax(input,axis = None)  指定某个维度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   #tf.cast(x,dtype) 对输入x进行类型转换
tf.summary.scalar('train_arruary',accuracy)

#merged = tf.summary.merge_all()      #合并所有的summary data
#write = tf.summary.FileWriter('log/',graph = tf.get_default_graph())   #tf.get_default_graph()得到默认的计算图
init = tf.global_variables_initializer()   #参数初始化
with tf.Session() as sess:
	init.run()
	step = 1

	test_num = test_image.shape[0]
	while step * batch_size < train_num:
		batch = fashion_mnist.train.next_batch(batch_size)
		batch_x = batch[0].reshape([batch_size,num_step,num_input])
		batch_y = batch[1]
		train_op.run(feed_dict={x:batch_x, y_:batch_y}) 
		if step % display_step == 0 or step == 1:
			train_accuracy =  accuracy.eval(feed_dict={x:batch_x, y_:batch_y})  
			train_loss = sess.run(loss, feed_dict={x:batch_x, y_:batch_y}) 
			#测试样本性能
			#test_accuracy = test_accuracy.eval()	
			print("step %d, training loss : %g, training accuracy %g" %(step, train_loss, train_accuracy))
			#rs = merged.eval(feed_dict={x:batch[0], y_:batch[1], keep_pro:0.5})   #tensor需要用eval，或者sess.run(mereged,...)
			#write.add_summary(rs,step)    #保存数据
		step+=1
	print("optimization finised!")
	#定义test准确率
	test_image_shaped = test_image.reshape([-1,num_step,num_input])
	print("test accuary %g" %accuracy.eval(feed_dict={x:test_image_shaped[:50,:,:], y_:test_labels[:50,:]}))

import os
import numpy as np
import tensorflow as tf
import input_data
import pandas as pd

# parameters
learning_rate = 0.001
# training_iters = 2000000
training_iters = 200000
batch_size = 32
display_step = 100

n_input = 7
n_output = 10
dropout = 0.9

def getTrainData():
	df = input_data.getTrainDataFrame()
	print "Train Data:", df.info()
	values = df.values
	np.random.shuffle(values)
	df = pd.DataFrame(data=values, columns=df.columns)
	label = df.label.values
	label = label.astype(np.int32)
	data = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']].values
	data = data / 255

	count = len(label)
	label1 = np.zeros((count, n_output))
	label1[np.arange(count), label] = 1
	# xunlianji
	label_train = label1[:count*0.7]
	data_train = data[:count*0.7]
	# yanzhengji
	label_vali = label1[count*0.7 : count*0.8]
	data_vali = data[count*0.7 : count*0.8]
	# ceshiji
	label_test = label1[count*0.8:]
	data_test = data[count*0.8:]
	
	return data_train, label_train, data_vali, label_vali, data_test, label_test


def getTestData():
	df = input_data.getTestDataFrame()
	print "Test Data:", df.info()
	values = df.values
	np.random.shuffle(values)
	df = pd.DataFrame(data=values, columns=df.columns)
	label = df.label.values
	label = label.astype(np.int32)
	data = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']].values
	data = data / 255

	count = len(label)
	label1 = np.zeros((count, n_output))
	label1[np.arange(count), label] = 1	
	return data, label1
	

weights = {
	'w1': tf.Variable(tf.random_normal([7, 128])),
	'w2': tf.Variable(tf.random_normal([128, 512])),
	'w3': tf.Variable(tf.random_normal([512, 64])),
	'w4': tf.Variable(tf.random_normal([64, 10]))
}

biases = {
	'b1': tf.Variable(tf.random_normal([128])),
	'b2': tf.Variable(tf.random_normal([512])),
	'b3': tf.Variable(tf.random_normal([64])),
	'b4': tf.Variable(tf.random_normal([10]))
}

def dnn(x, weights, biases, dropout):
	out1 = tf.nn.relu(tf.matmul(x, weights['w1']) + biases['b1'])
	out2 = tf.nn.relu(tf.matmul(out1, weights['w2']) + biases['b2'])
	# out2 = tf.nn.dropout(out2, dropout)
	out3 = tf.matmul(out2, weights['w3']) + biases['b3']
	out4 = tf.matmul(out3, weights['w4']) + biases['b4']
	return out4


def train():
	X = tf.placeholder(tf.float32, [None, n_input])
	Y = tf.placeholder(tf.float32, [None, n_output])
	P = dnn(X, weights, biases, dropout)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(P, Y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
	correct_pred = tf.equal(tf.argmax(P, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		train_data, train_label, vali_data, vali_label, test_data, test_label = getTrainData()
		# test_data, test_label = getTestData()
		j = 0
		while step * batch_size < training_iters:
			if j >= train_data.shape[0] / batch_size:
				j = 0
			batch_x = train_data[j * batch_size : (j + 1) * batch_size]
			batch_y = train_label[j * batch_size : (j + 1) * batch_size]
			sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
			if step % display_step == 0:
				loss, acc = sess.run([cost, accuracy], feed_dict={X:batch_x, Y:batch_y})
				vali_acc = sess.run(accuracy, feed_dict={X:vali_data, Y:vali_label})
				print 'Iter: ' + str(step * batch_size) + ', batch loss: ' + "{:.4f}".format(loss) + \
					', Training Acc: ' + "{:.4f}".format(acc) + ", Vali Acc: " + "{:.4f}".format(vali_acc)
			step += 1
			j += 1
		print 'Optimization Finished!'
		print 'Test Acc = ', sess.run(accuracy, feed_dict={X:test_data, Y:test_label})

train()
# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import gan
import cmtf.data.data_mnist as mnist


save_path = 'output/checkpoint.ckpt'
batch_size = 128
hidden_size = 128
learning_rate = 1e-2
outter_epoch = 10000
inner_epoch = 10


# check dir
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# model and data
model = gan.GAN(hidden_size, batch_size, learning_rate)
data = mnist.read_data_sets()


saver = tf.train.Saver()
if os.path.exists(save_path):
 	saver.restore(model.sess, save_path)
 	print 'restore'


for epoch in range(outter_epoch):
	loss_arr = []
	for _ in range(inner_epoch):
		x_, _ = data.train.next_batch(batch_size)
		loss = model.update_params(x_)
		loss_arr.append(loss)
	str_output = 'epoch: %d   loss: %.3f' %(epoch+1, np.mean(np.array(loss_arr)))
	if (epoch+1)%100 == 0:
		saver.save(model.sess, save_path)
		str_output += '   save'
	print str_output









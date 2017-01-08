# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import gan
from ImageOperation.images2one import *
from scipy import misc

save_path = 'output/checkpoint.ckpt'
batch_size = 128
hidden_size = 128
learning_rate = 1e-2
outter_epoch = 10000
inner_epoch = 10

# 模型
model = gan.GAN(hidden_size, batch_size, learning_rate)


# 生成图片
saver = tf.train.Saver()
saver.restore(model.sess, save_path)
sampled_images = model.sess.run(model.sampled_tensor, feed_dict={model.input_tensor: np.zeros([batch_size, 28*28])})
sampled_images = sampled_images.reshape(-1, 28, 28)


# 保存图片
img = images2one(sampled_images[:100])

if not os.path.exists('images'):
	os.makedirs('images')
misc.imsave('images/generate.png', img)

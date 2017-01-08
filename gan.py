# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope


class GAN(object):

	def __init__(self, hidden_size, batch_size, learning_rate):
		self.input_tensor = tf.placeholder(tf.float32, [None, 28 * 28])

		with arg_scope([layers.conv2d, layers.conv2d_transpose], 
		               activation_fn=concat_elu, 
		               normalizer_fn=layers.batch_norm, 
		               normalizer_params={'scale': True}):
			with tf.variable_scope("model"):
				D1 = discriminator(self.input_tensor)  # positive examples
				D_params_num = len(tf.trainable_variables())
				G = decoder(tf.random_normal([batch_size, hidden_size]))
				self.sampled_tensor = G

			with tf.variable_scope("model", reuse=True):
				D2 = discriminator(G)  # generated examples

		D_loss = self.__get_discrinator_loss(D1, D2)
		G_loss = self.__get_generator_loss(D2)

		params = tf.trainable_variables()
		D_params = params[:D_params_num]
		G_params = params[D_params_num:]

		global_step = tf.contrib.framework.get_or_create_global_step()
		self.train_discrimator = layers.optimize_loss(D_loss, global_step, learning_rate / 10, 'Adam', variables=D_params, update_ops=[])
		self.train_generator = layers.optimize_loss(G_loss, global_step, learning_rate, 'Adam', variables=G_params, update_ops=[])

		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())


	def __get_discrinator_loss(self, D1, D2):
		return (losses.sigmoid_cross_entropy(D1, tf.ones(tf.shape(D1))) + 
		        losses.sigmoid_cross_entropy(D2, tf.zeros(tf.shape(D1))))

	def __get_generator_loss(self, D2):
		return losses.sigmoid_cross_entropy(D2, tf.ones(tf.shape(D2)))

	def update_params(self, inputs):
		d_loss_value = self.sess.run(self.train_discrimator, {self.input_tensor: inputs})
		g_loss_value = self.sess.run(self.train_generator)
		return g_loss_value


def concat_elu(inputs):
	return tf.nn.elu(tf.concat(3, [-inputs, inputs]))


def encoder(input_tensor, output_size):
	net = tf.reshape(input_tensor, [-1, 28, 28, 1])
	net = layers.conv2d(net, 32, 5, stride=2)
	net = layers.conv2d(net, 64, 5, stride=2)
	net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
	net = layers.dropout(net, keep_prob=0.9)
	net = layers.flatten(net)
	return layers.fully_connected(net, output_size, activation_fn=None)


def discriminator(input_tensor):
	return encoder(input_tensor, 1)


def decoder(input_tensor):
	net = tf.expand_dims(input_tensor, 1)
	net = tf.expand_dims(net, 1)
	net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
	net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
	net = layers.conv2d_transpose(net, 32, 5, stride=2)
	net = layers.conv2d_transpose(
		net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
	net = layers.flatten(net)
	return net


if __name__ == '__main__':
	gan = GAN(10, 64, 0.1)







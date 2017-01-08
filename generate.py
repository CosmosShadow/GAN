# coding: utf-8

import os


imgs = self.sess.run(self.sampled_tensor)
for k in range(imgs.shape[0]):
	imgs_folder = os.path.join(directory, 'imgs')
	if not os.path.exists(imgs_folder):
		os.makedirs(imgs_folder)

	imsave(os.path.join(imgs_folder, '%d.png') % k, imgs[k].reshape(28, 28))

import scipy.sparse as sp
import numpy as np
import os.path
from PIL import Image
class Dataset(object):

    def __init__(self, path):
        pass
        self.input_croped,self.target_croped,self.num_train=read_data(path)
        self.epoch=0
        self.index_in_epoch=0
    def next_batch(self,batch_size):
        start=self.index_in_epoch
        self.index_in_epoch+=batch_size
        if self.index_in_epoch>self.num_train:
            self.epoch+=1
            perm=numpy.arange(self.num_train)
            num.random.shuffle(perm)
            self.input_croped=self.input_croped[perm]
            self.target_croped=self.target_croped[perm]
            start=0
            self.index_in_epoch=batch_size
        end=self.index_in_epoch
        return self.input_croped[start:end],self.target_croped[start:end]
    def read_data(self,path):
        pass
        lr_path=os.path.join(path,'lr.mat')
        hr_path=os.path.join(path,'hr.mat')
        lr=loadmat(lr_path)
        hr=loadmat(hr_path)
        print('read data complete.')
        lr_crop,hr_crop=crop(lr,hr)
        return lr_crop,hr_crop
    def crop(lr_in,hr_in)
        scale=2
        height=108
        width=108
        overlap=12
        hr_crop = tf.extract_image_patches(hr, [1, height, width, 1], [1, height - 2 * overlap, width - 2 * overlap, 1], [1, 1, 1, 1], padding='VALID')
        hr_reshape=tf.reshape(hr_crop, [tf.shape(hr_crop)[0] * tf.shape(hr_crop)[1] * tf.shape(hr_crop)[2], height, width, 3])
        lr_crop = tf.extract_image_patches(lr, [1, height/scale, width/scale, 1], [1, height/scale - 2 * overlap/scale, width/scale - 2 * overlap/scale, 1], [1, 1, 1, 1], padding='VALID')
        lr_reshape=tf.reshape(lr_crop, [tf.shape(lr_crop)[0] * tf.shape(lr_crop)[1] * tf.shape(lr_crop)[2], height/scale, width/scale, 3])
        sess = tf.Session()
        return sess.run([lr_reshape,hr_reshape],{lr:lr_in,hr:hr_in})


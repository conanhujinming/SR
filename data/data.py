import scipy.sparse as sp
import numpy as np
import os.path
from PIL import Image
import tensorflow as tf
class Dataset(object):

    def __init__(self, path):
        pass
        self.input_croped,self.target_croped,self.num_train=self.read_data(path)
        self.input_croped-=128
        self.epoch=0
        self.index_in_epoch=0
    def next_batch(self,batch_size):
        start=self.index_in_epoch
        self.index_in_epoch+=batch_size
        if self.index_in_epoch>self.num_train:
            self.epoch+=1
            perm=np.arange(self.num_train)
            num.random.shuffle(perm)
            self.input_croped=self.input_croped[perm]
            self.target_croped=self.target_croped[perm]
            start=0
            self.index_in_epoch=batch_size
        end=self.index_in_epoch
        return self.input_croped[start:end],self.target_croped[start:end]
    def read_data(self,path):
        pass
        lr_path=os.path.join(path,'lr.npy')
        hr_path=os.path.join(path,'hr.npy')
        #print lr_path
        lr=np.load(lr_path)
        hr=np.load(hr_path)
        print('read data complete.')
        return lr,hr,lr.shape[0]

    #used for testing
    def get_data(self):
        return self.input_croped,self.target_croped

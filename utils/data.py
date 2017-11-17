import scipy.sparse as sp
import numpy as np
import os.path
from PIL import Image
num=240
idx=0
hr=np.zeros(num*8,1000,1000,3)
lr=np.zeros(num*8,500,500,3)
#read_data
for parent,dirnames,filenames in os.walk(path):
for dirname in dirnames:
    image_name=os.path.join(parent,dirname)
        im=Image.open(image_name)
        im_flip=im.transpose(Image.FLIP_LEFT_RIGHT)
        data=np.array(im)
        data_flip=np.array(im_flip)
        lr[idx,:,:,:]=data
        idx+=1
        for i in xrange(3):
            data=np.rot90(data)
            lr[idx,:,:,:]=data
            idx+=1
            print idx
print('read data complete.')




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
        num=240
        idx=0
        hr=np.zeros(num*8,1000,1000,3)
        lr=np.zeros(num*8,500,500,3)
        #read_data
        for parent,dirnames,filenames in os.walk(path):
            for dirname in dirnames:
                image_name=os.path.join(parent,dirname)
                im=Image.open(image_name)
                im_flip=im.transpose(Image.FLIP_LEFT_RIGHT)
                data=np.array(im)
                data_flip=np.array(im_flip)
                lr[idx,:,:,:]=data
                idx+=1
                for i in xrange(3):
                    data=np.rot90(data)
                    lr[idx,:,:,:]=data
                    idx+=1

                print idx
        print('read data complete.')




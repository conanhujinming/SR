from PIL import Image
import numpy as np
import os
import os.path
rootdir='/home/cc/data/'
lr_path=os.path.join(rootdir,'small')
hr_path=os.path.join(rootdir,'big')
num=0
idx=0
for parent,dirnames,filenames in os.walk(lr_path):
    filenames.sort()
    for filename in filenames:
        image_name=os.path.join(parent,filename)
        #print image_name
        num+=1
    print num
#hr=np.zeros(num*8,1000,1000,3)
lr=np.zeros(num*8,500,500,3)
hr=np.zeros(num*8,1000,1000,3)
#read_data
for parent,dirnames,filenames in os.walk(lr_path):
for dirname in dirnames:
    image_name=os.path.join(parent,dirname)
        im=Image.open(image_name)
        im_flip=im.transpose(Image.FLIP_LEFT_RIGHT)
        data=np.array(im)
        data_flip=np.array(im_flip)
        lr[idx,:,:,:]=data
        idx+=1
        print idx
        for i in xrange(3):
            data=np.rot90(data)
            lr[idx,:,:,:]=data
            idx+=1
            print idx
        lr[idx,:,:,:]=data_flip
        idx+=1
        print idx
        for i in xrange(3):
            data_flip=np.rot90(data_flip)
            lr[idx,:,:,:]=data_flip
            idx+=1
            print idx
for parent,dirnames,filenames in os.walk(hr_path):
for dirname in dirnames:
    image_name=os.path.join(parent,dirname)
        im=Image.open(image_name)
        im_flip=im.transpose(Image.FLIP_LEFT_RIGHT)
        data=np.array(im)
        data_flip=np.array(im_flip)
        hr[idx,:,:,:]=data
        idx+=1
        print idx
        for i in xrange(3):
            data=np.rot90(data)
            hr[idx,:,:,:]=data
            idx+=1
            print idx
        hr[idx,:,:,:]=data_flip
        idx+=1
        print idx
        for i in xrange(3):
            data_flip=np.rot90(data_flip)
            hr[idx,:,:,:]=data_flip
            idx+=1
            print idx
np.save(rootdir,lr)
np.save(rootdir,hr)
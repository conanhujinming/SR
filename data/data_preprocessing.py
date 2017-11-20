from PIL import Image
import numpy as np
import os
import os.path
import tensorflow as tf
def crop(lr_in,hr_in):
    with tf.device('/cpu:0'):
        scale=2
        height=108
        width=108
        overlap=12
        hr=tf.placeholder(tf.float32,shape=[None,1024,1024,3])
        lr=tf.placeholder(tf.float32,shape=[None,512,512,3])
        print 'hello'
        hr_crop = tf.extract_image_patches(hr, [1, height, width, 1], [1, height - 2 * overlap, width - 2 * overlap, 1], [1, 1, 1, 1], padding='VALID')
        hr_reshape=tf.reshape(hr_crop, [tf.shape(hr_crop)[0] * tf.shape(hr_crop)[1] * tf.shape(hr_crop)[2], height, width, 3])
        print 'hi'
        lr_crop = tf.extract_image_patches(lr, [1, height/scale, width/scale, 1], [1, height/scale - 2 * overlap/scale, width/scale - 2 * overlap/scale, 1], [1, 1, 1, 1], padding='VALID')
        lr_reshape=tf.reshape(lr_crop, [tf.shape(lr_crop)[0] * tf.shape(lr_crop)[1] * tf.shape(lr_crop)[2], height/scale, width/scale, 3])
        sess = tf.Session()
        return sess.run([lr_reshape,hr_reshape],{lr:lr_in,hr:hr_in})

rootdir='/home/chenchen/data/dataset_scene'
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
lr=np.zeros([num*8,512,512,3],dtype=np.float32)
hr=np.zeros([num*8,1024,1024,3],dtype=np.float32)
#read_data
for parent,dirnames,filenames in os.walk(lr_path):
    filenames.sort()
    for filename in filenames:
        image_name=os.path.join(parent,filename)
        im=Image.open(image_name)
        im_flip=im.transpose(Image.FLIP_LEFT_RIGHT)
        data=np.array(im)
        data_flip=np.array(im_flip)
        lr[idx,:,:,:]=data
        idx+=1
        #print idx
        for i in xrange(3):
            data=np.rot90(data)
            lr[idx,:,:,:]=data
            idx+=1
            #print idx
        lr[idx,:,:,:]=data_flip
        idx+=1
        #print idx
        for i in xrange(3):
            data_flip=np.rot90(data_flip)
            lr[idx,:,:,:]=data_flip
            idx+=1
            #print idx
print idx
idx=0
for parent,dirnames,filenames in os.walk(hr_path):
    filenames.sort()
    for filename in filenames:
        image_name=os.path.join(parent,filename)
        im=Image.open(image_name)
        im_flip=im.transpose(Image.FLIP_LEFT_RIGHT)
        data=np.array(im)
        data_flip=np.array(im_flip)
        hr[idx,:,:,:]=data
        idx+=1
        #print idx
        for i in xrange(3):
            data=np.rot90(data)
            hr[idx,:,:,:]=data
            idx+=1
            #print idx
        hr[idx,:,:,:]=data_flip
        idx+=1
        #print idx
        for i in xrange(3):
            data_flip=np.rot90(data_flip)
            hr[idx,:,:,:]=data_flip
            idx+=1
            #print idx
print idx
#crop data
lr_crop,hr_crop=crop(lr,hr)
np.save(os.path.join(rootdir,'lr'),lr)
np.save(os.path.join(rootdir,'hr'),hr)

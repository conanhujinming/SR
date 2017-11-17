from PIL import Image
import numpy as np
import os
import os.path
num=0
idx=0
for parent,dirnames,filenames in os.walk(rootdir):
    filenames.sort()
    for filename in filenames:
        image_name=os.path.join(parent,filename)
        print image_name
        num+=1
#hr=np.zeros(num*8,1000,1000,3)
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
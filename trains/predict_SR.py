import tensorflow as tf
import numpy as np
import os,sys
import math

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(data_path)

from models.SR import SR
import utils.data
import utils.utils
flags=tf.app.flags

flags.DEFINE_string('test_data_in_path','data/Data/in.bmp','input data path')
flags.DEFINE_string('test_data_out_path','data/Data/out.bmp','output data path')
flags.DEFINE_integer('iterations',1000,'number of iterations')
flags.DEFINE_integer('batch_size','32','batch size')
flags.DEFINE_string('train_dir','ckpt/SR/','model save path')
flags.DEFINE_string('data_output_path','data/Output_data','output data path')
flags.DEFINE_integer('verbose',10,'show performance per X iterations')
flags.DEFINE_float('learning_rate','0.001','learning rate for training')
flags.DEFINE_string('optimizer','adam','specify an optimizer: adagrad, adam, rmsprop, sgd')
flags.DEFINE_integer('scale',2,'hr=lr*scale')

flags.DEFINE_integer('hidden_size',128,'hidden size')
flags.DEFINE_integer('bottleneck_size',64,'bottleneck size')

#input_data:lr_image
#target_data:hr_image
#ckpt_path:model save path
#optimizer:optimizer
#session:session
def create_model(ckpt_path,optimizer,session):
    model=SR(
        hidden_size=flags.hidden_size,
        bottleneck_size=flags.bottleneck_size,
        learning_rate=flags.learning_rate,
        optimizer=flags.optimizer,
        dtype=tf.float32,
        scope='SR',
        scale=flags.scale
        )

    ckpt=tf.train.get_checkpoint_state(ckpt_path)
    wrong=False
    if ckpt and ckpt.model_checkpoint_path:
        print('Reading model parameters from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('There is something wrong!')
        #session.run(tf.global_variables_initializer())
        wrong=True

    return model,wrong

def train():
    ckpt_path=flags.train_dir+'checkpoints/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    data=get_data(flags.test_data_path)
    #target_batch,input_batch=dataset.get_batch(flags.batch_size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model,wrong=create_model(ckpt_path,flags.optimizer,sess)
        if(wrong==True):
            return
        #_,training_loss=model.step(sess,input_batch,target_batch,training=True)
        prediction,_=model.step(sess,input_batch,target_batch,training=True)
        pass
        #TODO
        out_im = Image.fromarray(lr.astype(np.uint8))
        out_im.save(test_data_out_path)

#input an image dir
#output the array of the image
def get_data(path):
    pass
    im=im=Image.open(path)
    data=np.array(im)
    data-=0.5
    return data

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()


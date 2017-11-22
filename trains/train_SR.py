import tensorflow as tf
import numpy as np
import os,sys
import math

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)

from models.SR import SR
from utils import utils
from data import data
flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('data_path','/home/chenchen/data/dataset_scene/','input data path')
flags.DEFINE_integer('iterations',100000,'number of iterations')
flags.DEFINE_integer('batch_size','32','batch size')
flags.DEFINE_string('train_dir','../ckpt/SR/','model save path')
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
        hidden_size=FLAGS.hidden_size,
        bottleneck_size=FLAGS.bottleneck_size,
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        dtype=tf.float32,
        scope='SR',
        scale=FLAGS.scale
        )

    ckpt=tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Reading model parameters from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Creating model with fresh parameters.')
        session.run(tf.global_variables_initializer())

    return model

def train():
    ckpt_path=FLAGS.train_dir+'checkpoints/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    dataset=data.Dataset(FLAGS.data_path)
    #target_batch,input_batch=dataset.get_batch(FLAGS.batch_size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model=create_model(ckpt_path,FLAGS.optimizer,sess)
        itr_print=FLAGS.verbose
        itr_save=1000
        loss=0
        for itr in xrange(FLAGS.iterations):
            input_batch,target_batch=dataset.next_batch(FLAGS.batch_size)
            _,training_loss=model.step(sess,input_batch,target_batch,training=True)
            loss+=training_loss
            if(itr%itr_save==0):
                model.saver.save(sess,ckpt_path+'train',model.global_step)
            if(itr%itr_print==0):
                print 'Iteration:'+str(itr)+'Average loss:'+str(loss/itr_print)
                loss=0
            if(itr==(FLAGS.iterations-1) and itr%itr_print!=0):
                print 'Iteration:'+str(itr)+'Average loss:'+str(loss/(itr%itr_print))



def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()


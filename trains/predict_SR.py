import tensorflow as tf
import numpy as np
import os,sys
import math
from PIL import Image
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(data_path)
from utils import utils
from models.SR import SR
from utils import utils
from skimage.measure import compare_ssim as SSIM
flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('test_data_in_path','/home/chenchen/sr/my_sr/data/test/test.bmp','input data path')
flags.DEFINE_string('test_data_out_path','/home/chenchen/sr/my_sr/data/test/out.bmp','output data path')
flags.DEFINE_string('original_data_path','/home/chenchen/sr/my_sr/data/test/test_hr.bmp','original data')
#flags.DEFINE_integer('iterations',1000,'number of iterations')
#flags.DEFINE_integer('batch_size','32','batch size')
flags.DEFINE_string('train_dir','/home/chenchen/sr/my_sr/ckpt/SR/','model save path')
#flags.DEFINE_string('data_output_path','data/Output_data','output data path')
#flags.DEFINE_integer('verbose',10,'show performance per X iterations')
flags.DEFINE_float('learning_rate','0.001','learning rate for training')
flags.DEFINE_string('optimizer','adam','specify an optimizer: adagrad, adam, rmsprop, sgd')
flags.DEFINE_integer('scale',2,'hr=lr*scale')
flags.DEFINE_integer('hidden_size',128,'hidden size')
flags.DEFINE_integer('bottleneck_size',64,'bottleneck size')

#input an image dir
#output the array of the image
def get_data(path):
    pass
    im=Image.open(path)
    data=np.array(im)
    lr=np.zeros([1,512,512,3],dtype=np.float32)
    lr[0,:,:,:]=data

    return lr
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
    wrong=False
    if ckpt and ckpt.model_checkpoint_path:
        print('Reading model parameters from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('There is something wrong!')
        #session.run(tf.global_variables_initializer())
        wrong=True

    return model,wrong

#get PSNR of 2 image
def PSNR(img1, img2):
    """
    
    :param img1: numpy format for image
    :param img2: 
    :return: the psnr value of two images
    """
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def train():
    ckpt_path=FLAGS.train_dir
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    data=get_data(FLAGS.test_data_in_path)
    data-=128
    data=utils.padding(data)
    #target_batch,input_batch=dataset.get_batch(flags.batch_size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model,wrong=create_model(ckpt_path,FLAGS.optimizer,sess)
        if(wrong==True):
            return
        #_,training_loss=model.step(sess,input_batch,target_batch,training=True)
        prediction,_=model.step(sess,data,data,training=False)

    #print prediction
    prediction=np.reshape(prediction,prediction.shape[1:4])
    out_im = Image.fromarray(prediction.astype(np.uint8))
    out_im.save(FLAGS.test_data_out_path)

    #evalute
    im=Image.open(FLAGS.original_data_path)
    original=np.array(im,dtype=np.uint8)
    prediction=np.uint8(prediction)
    PSNR_score=PSNR(original,prediction)
    SSIM_score,_=SSIM(original, prediction, full=True, multichannel=True)
    print 'PSNR:'+str(PSNR_score)+' SSIM:'+str(SSIM_score)

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()


import tensorflow as tf
from utils import utils
class SR(object):
    def __init__(self,
                 hidden_size,
                 bottleneck_size,
                 learning_rate,
                 optimizer='adam',
                 dtype=tf.float32,
                 scope='SR',
                 scale=2
                ):
        self.hidden_size=hidden_size
        self.bottleneck_size=bottleneck_size
        self.learning_rate=learning_rate
        self.optimizer=optimizer
        self.dtype=dtype
        self.scale=scale
        with tf.variable_scope(scope):
            self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=self.dtype, name='learning_rate')
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

            self.build_graph()

            self.saver = tf.train.Saver(max_to_keep=10)

    def build_graph(self):
        self._create_placeholder()
        self._create_loss()
        self._create_optimizer()

    def _create_placeholder(self):
        pass
        self.input=tf.placeholder(self.dtype,[None,None,None,3],name='input')
        self.target=tf.placeholder(self.dtype,[None,None,None,3],name='output')
    
    def _create_loss(self):
        pass
        x=tf.layers.conv2d(self.input,self.hidden_size,1,activation=None,name='in')
        #self.test=tf.reduce_mean(x)
        #low resolution
        for i in range(6):
            x=utils.crop_by_pixel(x,1)+self.conv(x,self.hidden_size,self.bottleneck_size,'lr_conv'+str(i))
        temp=tf.nn.relu(x)
        #up sampling
        x=tf.image.resize_nearest_neighbor(x,tf.shape(x)[1:3]*2)+tf.layers.conv2d_transpose(temp,self.hidden_size,2,strides=2,name='up_sampling')

        #high resolution
        for i in range(4):
            x=utils.crop_by_pixel(x,1)+self.conv(x,self.hidden_size,self.bottleneck_size,'hr_conv'+str(i))
        x=tf.nn.relu(x)
        self.prediction=tf.layers.conv2d(x,3,1,name='out')
        self.target_crop=utils.crop_center(self.target,tf.shape(self.prediction)[1:3])
        #self.test=tf.reduce_mean(self.prediction)
        #self.test=tf.reduce_mean(self.prediction)
        self.loss = tf.losses.mean_squared_error(self.target_crop, self.prediction)
        #self.loss = tf.losses.log_loss(self.target_crop, self.prediction)
        #self.loss = tf.losses.absolute_difference(self.target_crop, self.prediction)
    def _create_optimizer(self):
        pass
        #you can put more optimizer here
        if(self.optimizer=='adam'):
            optimizer=tf.train.AdamOptimizer(self.learning_rate)
        self.updates=optimizer.minimize(self.loss,self.global_step)

    def conv(self,x,hidden_size,bottleneck_size,name,with_last_relu=True):
        x=tf.nn.relu(x)
        x=tf.layers.conv2d(x,bottleneck_size,1,activation=tf.nn.relu,name=name+'_proj')
        x=tf.layers.conv2d(x,hidden_size,3,activation=None,name=name+'_filt')
        return x

    def step(self,session,input,target,training=False):
        input_feed={}
        input_feed[self.input.name]=input
        input_feed[self.target.name]=target
        if training:
            output_feed=[self.prediction,self.loss,self.updates]
        else:
            output_feed=[self.prediction]

        outputs=session.run(output_feed,input_feed)
        if(len(outputs)==1):
            outputs=[outputs[0],None]
        return outputs[0],outputs[1]

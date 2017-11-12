import tensorflow as tf
class SR(object):
	def __init__(self,
				 hidden_size,
				 bottleneck_size,
				 learning_rate,
				 learning_rate_decay_factor,
				 optimizer='adam',
				 dtype=tf.float32,
				 scope='SR'
				):
		self.hidden_size=hidden_size
		self.bottleneck_size=bottleneck_size
		self.learning_rate=learning_rate
		self.learning_rate_decay_factor=learning_rate_decay_factor
		self.optimizer=optimizer
		self.dtype=dtype

		with tf.variable_scope(scope):
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype, name='learning_rate')
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

            self.build_graph()

            self.saver = tf.train.Saver(max_to_keep=10)

    def build_graph(self,input):
    	self._create_placeholder()
        self._create_loss(input)
        self._create_optimizer()

    def _create_placeholder(self):
    	pass

    def _create_loss(self,input):
    	pass
    	x=tf.layers.conv2d(input,hidden_size,1,activation=None,name='input')
    	
    	#low resolution
    	for i in range(6)
    		x=x+conv(x,self.hidden_size,bottleneck_size,'lr_conv'+str(i))

    	#up sampling
    	temp=tf.nn.relu(x)
    	x=tf.image.resize_nearest_neighbor(x,tf.shape(x)[1:3]*2)+tf.layers.conv2d_transpose(temp,hidden_size,2,strides=2,name='up_sampling')

    	#high resolution



   	def _create_optimizer(self):
   		pass

   	def conv(x,hidden_size,bottleneck_size,name):
   		x=tf.nn.relu(x)
   		x=tf.layers.conv2d(x,bottleneck_size,1,activation=tf.nn.relu,name=name+'_proj')
   		x=tf.layers.conv2d(x,hidden_size,3,activation=None,name=name+'_filt',padding='same')

   		return x

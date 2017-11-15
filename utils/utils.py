import tensorflow as tf

#crop num pixels from each side
def crop_by_pixel(x,num):
	shape=tf.shape(x)[1:3]
	return tf.slice(x,[0,num,num,0],[-1,shape[0]-2*num,shape[1]-2*num,-1])
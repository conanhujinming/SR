import tensorflow as tf

#crop num pixels from each side
def crop_by_pixel(x,num):
	shape=tf.shape(x)[1:3]
	return tf.slice(x,[0,num,num,0],[-1,shape[0]-2*num,shape[1]-2*num,-1])

def crop_center(image, target_shape):
    origin_shape = tf.shape(image)[1:3]
    return tf.slice(image, [0, (origin_shape[0] - target_shape[0]) / 2, (origin_shape[1] - target_shape[1]) / 2, 0], [-1, target_shape[0], target_shape[1], -1])
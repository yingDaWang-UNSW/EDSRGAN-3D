#import keras as K
import tensorflow as tf
from tensorlayer.layers import *
import numpy as np
def smooth_gan_labels(y):
    if y == 0:
        y_out = tf.random_uniform(shape=y.get_shape(), minval=0.0, maxval=0.3)
    else:
        y_out = tf.random_uniform(shape=y.get_shape(), minval=0.7, maxval=1.2)
    return y_out
    
#import subprocess
#gnuplot = subprocess.Popen(["/usr/bin/gnuplot"], 
#                           stdin=subprocess.PIPE)
#def plotTerminal(x, y):
#    gnuplot.stdin.write(b"set term dumb 100 25\n")
#    gnuplot.stdin.write(b"plot '-' using 1:2 title 'Line1' with linespoints \n")
#    for i,j in zip(x,y):
#       gnuplot.stdin.write(b"%f %f\n" % (i,j))
#    gnuplot.stdin.write(b"e\n")
#    gnuplot.stdin.flush()

def subPixelConv3d(net, img_width, img_height, img_depth, stepsToEnd, n_out_channel):
    i = net

    r = 2
    a = (img_width // (2 * stepsToEnd))
    b = (img_height // (2 * stepsToEnd))
    z = (img_depth // (2 * stepsToEnd))
    c = tf.shape(i)[3]
    bsize = tf.shape(i)[0]  # Handling Dimension(None) type for undefined batch dim
    xs = tf.split(i, r, 4)  # b*h*w*d*r*r*r
    xr = tf.concat(xs, 3)  # b*h*w*(r*d)*r*r
    xss = tf.split(xr, r, 4)  # b*h*w*(r*d)*r*r
    xrr = tf.concat(xss, 2)  # b*h*(r*w)*(r*d)*r
    x = tf.reshape(xrr, (bsize, r * a, r * b, r * z, n_out_channel))  # b*(r*h)*(r*w)*(r*d)*n_out n_out=64/2^

    return x

# The implementation of PixelShuffler
def pixelShuffler3D(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    l = size[3]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, l, scale, scale, scale]
    shape_2 = [batch_size, h * scale, w * scale, l*scale, 1]
    # split the tensor based on the scale factor
    xs = tf.split(inputs, scale, 4)  # b*h*w*d*r*r*r
    xr = tf.concat(xs, 3)  # b*h*w*(r*d)*r*r
    xss = tf.split(xr, scale, 4)  # b*h*w*(r*d)*r*r
    xrr = tf.concat(xss, 2)  # b*h*(r*w)*(r*d)*r
    
    #xsss = tf.split(xrr, scale, 4) 
    #xrrr = tf.concat(xsss, 1) 
    
    output = tf.reshape(xrr, (batch_size, h * scale, w * scale, l*scale, channel_target)) 
    #output=xrrr
    # Reshape and transpose for periodic shuffling for each channel
    #input_split = tf.split(inputs, channel_target, axis=4)
    #temp=[phaseShift(x, scale, shape_1, shape_2) for x in input_split]
    #output = tf.concat(temp, axis=4)
    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 4, 2, 5, 3, 6])

    return tf.reshape(X, shape_2)

def depth_to_space3D(x, block_size):
    #x = np.asarray(x)
    
    batch = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    length = tf.shape(x)[3]
    channel =  x.get_shape().as_list()[-1]

    
    y = tf.reshape(x, (batch, height, block_size, width, block_size, length, block_size, channel//(block_size**3)))
#    shuf=np.arange(5)+1
#    np.random.shuffle(shuf)
#    print(shuf)
    z = tf.transpose(y, perm=[0,1,4,2,5,3,6,7])
    
    z = tf.reshape(z, (batch, height*block_size, width*block_size, length*block_size, channel//(block_size**3)))
    return z


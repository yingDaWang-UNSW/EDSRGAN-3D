#import keras as K
import tensorflow as tf
#from utils import subPixelConv3d, pixelShuffler3D, phaseShift, depth_to_space3D
from tensorflow.keras.layers import UpSampling3D
from tensorlayer.layers import *
import numpy as np
import tensorflow.contrib.slim as slim

'''
########################################################################
GENERATORS
########################################################################
'''

def lrelu1(x):
    return tf.maximum(x, 0.2 * x)

def lrelu2(x):
    return tf.maximum(x, 0.3 * x)
    
def prelu(_x):
#    alphas = tf.get_variable('SRGAN_g/alpha', _x.get_shape()[-1],
#                       initializer=tf.constant_initializer(0.0),
#                        dtype=tf.float32)
#    pos = tf.nn.relu(_x)
#    neg = alphas * (_x - abs(_x)) * 0.5
#    
#    _x=pos + neg
    _x=tf.keras.layers.PReLU(shared_axes=[1, 2, 3])(_x)
    return _x

def relu(x):
    return tf.maximum(x, 0)

def activate(x, actFunc):
    if actFunc=='lrelu':
        x=lrelu1(x)
    elif actFunc=='relu':
        x=relu(x)
    elif actFunc=='prelu':
        x=prelu(x)
    return x
    
def _variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var
    
def conv3dydw(x, numInFilters, numFilters, kernelSize, stride, name, reuse, trainable, padding, use_bias, kernel_initializer):

    #x = tf.layers.conv3d(x, numFilters, kernelSize, (stride, stride, stride), name=name, reuse=reuse, trainable=trainable, padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer)
    
    x =  tf.keras.layers.Conv3D(filters=numFilters, kernel_size=kernelSize, strides=stride, padding=padding)(x)
    
    #weights = _variable(name+'weights', shape=[kernelSize,kernelSize, kernelSize,numInFilters,numFilters],initializer=tf.contrib.layers.xavier_initializer())
    #biases = _variable(name+'biases',[numFilters],initializer=tf.contrib.layers.xavier_initializer())

    #x = tf.nn.conv3d(x, weights, strides=[1, stride, stride, stride, 1], padding='SAME')
    #x = tf.nn.bias_add(x, biases)
    return x

def gatedTFactivation3D(x):
    x_2_1, x_2_2 = tf.split(axis=4,num_or_size_splits=2,value=x)
    x = tf.nn.tanh(x_2_1) * tf.nn.sigmoid(x_2_2)
    return x




'''

EDSR - PRELU

'''
#edsr-like generator built on tensorflow
def generatorTF(inputs, kernelSize=3, numResidualBlocks=16, numFilters=64, scaleFactor=4, reuse=False, isTrain=True):
    stride=1
    biasFlag=True
    kernelInit=None
    activation='prelu'
    #the residual block structure
    def residual_block(inputs, numFilters, kernelSize, stride, scope):
        with tf.variable_scope(scope):
            net = conv3dydw(inputs, numFilters, numFilters, kernelSize, stride, name='Conv1', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            net = activate(net, activation)
            
            net = conv3dydw(net, numFilters, numFilters, kernelSize, stride, name='Conv2', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            net = net + inputs
        return net
        
    with tf.variable_scope('SRGAN_g', reuse=reuse):
        #input convolution
        with tf.variable_scope('initialFilter'):
            x = inputs
            #x=tf.keras.backend.in_train_phase(inputs, tf.pad(inputs, tf.constant([[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]]), 'SYMMETRIC'))

            x = tf.keras.layers.Conv3D(input_shape = (None, None, None, None, 1), filters=numFilters, kernel_size=kernelSize, strides=stride, padding='same')(x)
            #x = conv3dydw(x, numFilters, kernelSize, stride, name='convInit', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            #x.uses_learning_phase = True
            
        shallowOutput = x
        
        #residual blocks
        for i in range(1, numResidualBlocks+1, 1):
            scope='resBlock_%d'%(i)
            x = residual_block(x, numFilters, kernelSize, stride, scope)
        
        #output convolution
        with tf.variable_scope('resBlockOutFilt'):
            x = conv3dydw(x, numFilters, numFilters, kernelSize, stride, name='convOut', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
        
        #Skip connection
        x = x + shallowOutput
        
        factor=1
        #subpixelconvolution
        with tf.variable_scope('subpixelconv_stage1'):
            x = conv3dydw(x, numFilters, numFilters*factor, kernelSize, stride, name='convSub1', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            x = activate(x, activation)
            
            #x = depth_to_space3D(x,2)
            #x = pixelShuffler3D(x,2)
            x = tf.keras.layers.UpSampling3D(name='UpSampling3D_1')(x)
        with tf.variable_scope('subpixelconv_stage2'):
            x = conv3dydw(x, numFilters, numFilters*factor, kernelSize, stride, name='convSub2', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            x = activate(x, activation)
            
            #x = depth_to_space3D(x,2)
            #x = pixelShuffler3D(x,2)
            x = tf.keras.layers.UpSampling3D(name='UpSampling3D_2')(x)
        # output channels
        with tf.variable_scope('output_stage'):
            x = conv3dydw(x, numFilters, 1, kernelSize, stride, name='convLast', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            x=tf.nn.tanh(x)
    return x

'''

WDSR - PRELU

'''

def generatorTFWide(inputs, kernelSize=3, numResidualBlocks=16, numFilters=64, scaleFactor=4, reuse=False, isTrain=True):
    stride=1
    biasFlag=True
    kernelInit=None
    activation='prelu'
    #the residual block structure
    def residual_block(inputs, numFilters, kernelSize, stride, scope):
        with tf.variable_scope(scope):
            net = conv3dydw(inputs, numFilters, numFilters*6, 1, stride, name='Conv1', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            net = activate(net, activation)
            
            net = conv3dydw(net , numFilters*6, int(numFilters*0.8), 1, stride, name='Conv2', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            net = conv3dydw(net, int(numFilters*0.8), numFilters, kernelSize, stride, name='Conv3', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            net = net + inputs
        return net
        
    with tf.variable_scope('SRGAN_g', reuse=reuse):
        #input convolution
        with tf.variable_scope('initialFilter'):
            x = inputs
            #x = tf.keras.backend.in_train_phase(x, tf.pad(inputs, tf.constant([[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]]), 'SYMMETRIC'))
            #x = tf.pad(x, tf.constant([[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]]), 'SYMMETRIC')
            #x.uses_learning_phase = True
            x = tf.keras.layers.Conv3D(input_shape = (None, None, None, None, 1), filters=numFilters, kernel_size=3, strides=stride, padding='same')(x)
            #x = conv3dydw(x, numFilters, kernelSize, stride, name='convInit', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
        shallowOutput=x
        
        #residual blocks
        for i in range(1, numResidualBlocks+1, 1):
            scope='resBlock_%d'%(i)
            x = residual_block(x, numFilters, kernelSize, stride, scope)
        
        #output convolution
#        with tf.variable_scope('resBlockOutFilt'):
#            x = conv3dydw(x, numFilters, kernelSize, stride, name='convOut', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
        
        # CONVOLVE IN PARALLEL
        deepOutput = x
        
        #subpixelconvolution
        with tf.variable_scope('subpixelconv_shallow'):
            x = conv3dydw(shallowOutput, numFilters, numFilters, kernelSize, stride, name='convSub1', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            x = activate(x, activation)
            
            #x = depth_to_space3D(x,2)
            x = tf.keras.layers.UpSampling3D(name='UpSampling3D_2')(x)
        with tf.variable_scope('subpixelconv_shallow2'):
            x = conv3dydw(x, numFilters, numFilters, kernelSize, stride, name='convSub1', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
           
            x = activate(x, activation)
           
            #shallowOutput = depth_to_space3D(x,2)
            shallowOutput = tf.keras.layers.UpSampling3D(name='UpSampling3D_2')(x)
        
        with tf.variable_scope('subpixelconv_deep'):
            x = conv3dydw(deepOutput, numFilters, numFilters, 5, stride, name='convSub1', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            x = activate(x, activation)
            
            #x = depth_to_space3D(x,2)
            x = tf.keras.layers.UpSampling3D(name='UpSampling3D_2')(x)
        with tf.variable_scope('subpixelconv_deep2'):
            x = conv3dydw(x, numFilters, numFilters, 5, stride, name='convSub1', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            x = activate(x, activation)
            
            #deepOutput = depth_to_space3D(x,2)
            deepOutput = tf.keras.layers.UpSampling3D(name='UpSampling3D_2')(x)
        
        x = deepOutput + shallowOutput
        # output channels using nin (faster)
        with tf.variable_scope('output_stage'):
            x = conv3dydw(x, numFilters, 1, 1, stride, name='convLast', reuse=reuse, trainable=isTrain, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            
            x=tf.nn.tanh(x)
    return x


'''

SRCNN

'''

# TODO: I have no idea how tf TF seems to outperform TL
def generator(input_gen, kernel, nb, scaleFactor, reuse, numFilters):

    # stack the model layers
    with tf.variable_scope("SRGAN_g", reuse=reuse):

        # pass tensorflow to tensorlayer
        x = InputLayer(input_gen, name='Input Tensor')
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 1, numFilters], strides=[1, 1, 1, 1, 1], padding='SAME', name='Conv1')

        inputRB = x
        inputadd = x

        # stack the residual blocks
        for i in range(nb):
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, numFilters], strides=[1, 1, 1, 1, 1], padding='SAME',act=lrelu1, name='ResBlock/%sConvA' % i)
            
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, numFilters], strides=[1, 1, 1, 1, 1], padding='SAME', name='ResBlock/%sConvB' % i)
            
            # short skip connection
            x = ElementwiseLayer([x, inputadd], tf.add, name='ResBlock/%sAdditionLayer' % i)
            inputadd = x

        # large skip connection
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, numFilters], strides=[1, 1, 1, 1, 1], padding='SAME', name='Conv2')
        
        x = ElementwiseLayer([x, inputRB], tf.add, name='Conv2AdditionLayer')

        # ____________RC______________#
        factor=1
        # upscaling block 1
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, numFilters*factor], act=lrelu1, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv1-ub/1')
        
        #x = pixelShuffler3D(x, 2)
        x = UpSampling3D(name='UpSampling3D_1')(x.outputs)
        
        x = Conv3dLayer(InputLayer(x, name='in ub1 conv2'), shape=[kernel, kernel, kernel, numFilters*factor, numFilters*factor], act=lrelu1, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv2-ub/1')

        # upscaling block 2
        #x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, numFilters], act=lrelu1, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv1-ub/2')
        #x = pixelShuffler3D(x.outputs, 2)
        x = UpSampling3D(name='UpSampling3D_2')(x.outputs)
        #x = Conv3dLayer(InputLayer(x, name='in ub2 conv2'), shape=[kernel, kernel, kernel, numFilters, numFilters], act=lrelu1, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv2-ub/2')

        x = Conv3dLayer(InputLayer(x, name='in ub2 conv2'), shape=[kernel, kernel, kernel, numFilters*factor, 1], strides=[1, 1, 1, 1, 1], act=tf.nn.tanh, padding='SAME', name='convlast')
            
    return x.outputs



# TODO: remove batchnorm layers and add a shuffler layer
def generatorMS(input_gen, kernel, reuse,nb, numFilters, is_train=True):
    # stack the model layers
    with tf.variable_scope("SRGAN_g", reuse=reuse):

        # pass tensorflow to tensorlayer
        x = InputLayer(input_gen, name='Input Tensor')
        
        inputImage = x
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, 1, numFilters], strides=[1, 1, 1, 1, 1], padding='SAME', name='Conv1')
        
        x = BatchNormLayer(x, act=lrelu1, is_train=is_train, name='BatchNormConv1')
        
        #keep the skip connections for now
        inputRB = x
        inputadd = x

        # stack the residual blocks
        for i in range(nb):
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, numFilters], strides=[1, 1, 1, 1, 1], padding='SAME', name='ResBlock/%sConvA' % i)
            
            x = BatchNormLayer(x, act=lrelu1, is_train=is_train, name='ResBlock/%sBatchNormA' % i)
            
            x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, numFilters], strides=[1, 1, 1, 1, 1], padding='SAME', name='ResBlock/%sConvB' % i)
            
            x = BatchNormLayer(x, is_train=is_train, name='ResBlock/%sBatchNormB' % i, )
            # short skip connection
            x = ElementwiseLayer([x, inputadd], tf.add, name='ResBlock/%sAdditionLayer' % i)
            inputadd = x

        # large skip connection
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, numFilters], strides=[1, 1, 1, 1, 1], padding='SAME', name='Conv2')
        
        x = BatchNormLayer(x, is_train=is_train, name='BatchNormConv2')
        
        x = ElementwiseLayer([x, inputRB], tf.add, name='Conv2AdditionLayer')
        x = ElementwiseLayer([x, inputImage], tf.add, name='SuperpositionLayer')
        # modify volume
        # deal with this later
        
        
        # residual Addition to LR

        #x = DeConv3dLayer(x, shape=[kernel * 2, kernel * 2, kernel * 2, 64, numFilters], act=lrelu1, strides=[1, 2, 2, 2, 1], output_shape=[tf.shape(input_gen)[0], tf.shape(input_gen)[1], tf.shape(input_gen)[2], tf.shape(input_gen)[3], 64], padding='SAME', name='conv1-ub-subpixelnn/1')
        
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, numFilters, 1], strides=[1, 1, 1, 1, 1], act=tf.nn.tanh, padding='SAME', name='convlast')

    return x

'''
########################################################################
DISCRIMINATORS
########################################################################
'''

def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)

# Our dense layer
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise
    
def discriminatorTF(input_disc, kernel, is_train=True, reuse=False):

    stride=1
    biasFlag=True
    kernelInit=None
    
    def discriminator_block(inputs, numInFilters, numFilters, kernelSize, stride, scope, reuse):
        with tf.variable_scope(scope):
            net = conv3dydw(inputs, numInFilters, numFilters, kernelSize, stride, name='DConv1', reuse=reuse, trainable=is_train, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            net = batchnorm(net, is_train)
            net = lrelu2(net)
        return net
    
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        x = input_disc
        # cripple the disriminator with noise
        #x = gaussian_noise_layer(x, 0)
        x =  tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=stride, padding='SAME')(x)
        x = lrelu2(x)
        # The discriminator block part
        # block 1
        x = discriminator_block(x, 64, 64, 3, 2, 'disblock_1', reuse)

        # block 2
        x = discriminator_block(x, 64, 128, 3, 1, 'disblock_2', reuse)

        # block 3
        x = discriminator_block(x, 128, 128, 3, 2, 'disblock_3', reuse)

        # block 4
        x = discriminator_block(x, 128, 256, 3, 1, 'disblock_4', reuse)

        # block 5
        x = discriminator_block(x, 256, 256, 3, 2, 'disblock_5', reuse)

        # block 6
        x = discriminator_block(x, 256, 512, 3, 1, 'disblock_6', reuse)

        # block_7
        x = discriminator_block(x, 512, 512, 3, 2, 'disblock_7', reuse)

        # The dense layer 1
        with tf.variable_scope('dense_layer_1'):
            x = slim.flatten(x)
            x = denselayer(x, 1024)
            x = lrelu2(x)

        # The dense layer 2
        with tf.variable_scope('dense_layer_2'):
            x = denselayer(x, 1)
            logits = x
            x = tf.nn.sigmoid(x)
    return x, logits


def discriminator(input_disc, kernel, is_train=True, reuse=False):
    
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        df_dim=64
        x = InputLayer(input_disc, name='in')
        
        x = Conv3dLayer(x, act=lrelu2, shape=[kernel, kernel, kernel, 1, df_dim], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv1')
        
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, df_dim, df_dim], strides=[1, 2, 2, 2, 1], padding='SAME', name='conv2')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv2', act=lrelu2)

        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, df_dim, df_dim*2], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv3', act=lrelu2)
        
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, df_dim*2, df_dim*2], strides=[1, 2, 2, 2, 1], padding='SAME', name='conv4')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv4', act=lrelu2)

        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, df_dim*2, df_dim*4], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv5', act=lrelu2)
        
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, df_dim*4, df_dim*4], strides=[1, 2, 2, 2, 1], padding='SAME', name='conv6')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv6', act=lrelu2)

        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, df_dim*4, df_dim*8], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv7')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv7', act=lrelu2)
        
        x = Conv3dLayer(x, shape=[kernel, kernel, kernel, df_dim*8, df_dim*8], strides=[1, 2, 2, 2, 1], padding='SAME', name='conv8')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv8', act=lrelu2)
        
        x = FlattenLayer(x, name='flatten')
        x = DenseLayer(x, n_units=1024, act=lrelu2, name='dense1')
        x = DenseLayer(x, n_units=1, name='dense2')

        logits = x.outputs
        x.outputs = tf.nn.sigmoid(x.outputs, name='output')

        return x.outputs, logits
        


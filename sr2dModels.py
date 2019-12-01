import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def gated(x):
    x=x.outputs
    x_2_1, x_2_2 = tf.split(axis=3,num_or_size_splits=2,value=x)
    x = tf.nn.tanh(x_2_1) * tf.nn.sigmoid(x_2_2)
    x = InputLayer(x)
    return x

def lrelu(x):
    return tf.maximum(x, 0.2 * x)
    
def prelu(_x):
#    alphas = tf.get_variable('SRGAN_g/alpha', _x.get_shape()[-1],
#                       initializer=tf.constant_initializer(0.0),
#                        dtype=tf.float32)
#    pos = tf.nn.relu(_x)
#    neg = alphas * (_x - abs(_x)) * 0.5
#    
#    _x=pos + neg
    _x=tf.keras.layers.PReLU(shared_axes=[1, 2])(_x)
    return _x

def relu(x):
    return tf.maximum(x, 0)

def activate(x, actFunc):
    if actFunc=='lrelu':
        x=lrelu(x)
    elif actFunc=='relu':
        x=relu(x)
    elif actFunc=='prelu':
        x=prelu(x)
    elif actFunc=='gated':
        x=gated(x)
    return x
    
#a hyper skip connection, but keep it fully convolutional, make it wdsrb-like
def neuralSkip(x,y, numFilters, kernelSize, stride, actFunc, w_init, b_init, name):
    m = Conv2d(x, numFilters//2, (1, 1), (stride, stride), act=actFunc, padding='SAME', W_init=w_init, b_init=b_init, name=f'k{kernelSize}n{numFilters}s{stride}/c1/nskip{name}')
    m = Conv2d(m, numFilters*2, (1, 1), (stride, stride), act=None, padding='SAME', W_init=w_init, b_init=b_init, name=f'k{kernelSize}n{numFilters}s{stride}/c2/nskip{name}')
    m = Conv2d(m, numFilters, (kernelSize, kernelSize), (stride, stride), act=tf.nn.tanh, padding='SAME', W_init=w_init, name=f'k{kernelSize}n{numFilters}s{stride}/c3/nskip{name}')
    #y = y+m*x
    y = ElementwiseLayer([y, ElementwiseLayer([m ,x], tf.math.multiply, name='nskipgrad')], tf.add, name='nskipbias')
    return y

#the WDSR-like skip with subpixel
def subPixelSkip(shallow, deep, numFilters, kernelSize, stride, actFunc, w_init, b_init, name):
    shallow = Conv2d(shallow, numFilters, (kernelSize, kernelSize), (stride, stride), act=actFunc, padding='SAME', W_init=w_init, b_init=b_init, name=f'k{kernelSize}n{numFilters}s{stride}/c1/nskip{name}')
    deep = ElementwiseLayer([deep, shallow], tf.add, name='subPixelSkip')
    return deep

# EDSR and SR-Resnet Residual Blocks, activate batchnorm if needed
def residualBlock(n, numFilters, kernelSize, stride, batchNorm, actFunc, w_init, b_init, g_init, i):
    nn = Conv2d(n, numFilters, (kernelSize, kernelSize), (stride, stride), act=actFunc, padding='SAME', W_init=w_init, b_init=b_init, name=f'k{kernelSize}n{numFilters}s{stride}/c1/{i}')
    nn = BatchNormLayer(nn, act=actFunc, is_train=batchNorm, gamma_init=g_init, name=f'k{kernelSize}n{numFilters}s{stride}/b1/{i}')
    nn = Conv2d(nn, numFilters, (kernelSize, kernelSize), (stride, stride), act=None, padding='SAME', W_init=w_init, b_init=b_init, name=f'k{kernelSize}n{numFilters}s{stride}/c2/{i}')
    nn = BatchNormLayer(nn, is_train=batchNorm, gamma_init=g_init, name=f'k{kernelSize}n{numFilters}s{stride}/b2/{i}')
    n = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
    #n = neuralSkip(n, nn, numFilters, kernelSize, stride, actFunc, w_init, b_init, f'{i}')
    return n

def generator2D(t_image, scaleFactor, numResBlocks, numFilters, kernelSize, initKernelFactor, activation, batchNorm=True, reuse=False):
    """ SRGAN generator"""
    stride=1
    actFunc = lambda x: activate(x, activation)
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, numFilters, (kernelSize*initKernelFactor, kernelSize*initKernelFactor), (1, 1), act=None, padding='SAME', W_init=w_init, name=f'k{kernelSize*initKernelFactor}n{numFilters}s{stride}/cInit')
        shallow = n
        # B residual blocks
        for i in range(numResBlocks):
            n = residualBlock(n, numFilters, kernelSize, stride, batchNorm, actFunc, w_init, b_init, g_init, i)
        # redundant output conv
        n = Conv2d(n, numFilters, (kernelSize, kernelSize), (stride, stride), act=None, padding='SAME', W_init=w_init, b_init=b_init, name=f'k{kernelSize}n{numFilters}s{stride}/c/m')
        n = BatchNormLayer(n, is_train=batchNorm, gamma_init=g_init, name=f'k{kernelSize}n{numFilters}s{stride}/b/m')
        n = ElementwiseLayer([n, shallow], tf.add, name='add3')
        #n = neuralSkip(shallow, n, numFilters, kernelSize, stride, actFunc, w_init, b_init, f'deep')
        
        # start upscaling layers
        subScale=scaleFactor//2
        subPixConvFilters=numFilters*(subScale)**2
        n = Conv2d(n, subPixConvFilters, (kernelSize, kernelSize), (stride, stride), act=None, padding='SAME', W_init=w_init, name=f'k{kernelSize}n{subPixConvFilters}s{stride}/1')
        n = SubpixelConv2d(n, scale=subScale, n_out_channel=None, act=actFunc, name=f'pixelshufflerx{subScale}/1')
        # add a subpixelSkip here
        # n = subPixelSkip(t_image, n, subPixConvFilters, 5, stride, actFunc, w_init, b_init, f'subSkip1')
        n = Conv2d(n, subPixConvFilters, (kernelSize, kernelSize), (stride, stride), act=None, padding='SAME', W_init=w_init, name=f'k{kernelSize}n{subPixConvFilters}s{stride}/2')
        n = SubpixelConv2d(n, scale=subScale, n_out_channel=None, act=actFunc, name=f'pixelshufflerx{subScale}/2')
        # add a subpixelSkip here
        # n = subPixelSkip(shallow, n, subPixConvFilters, 5, stride, actFunc, w_init, b_init, f'subSkip2')
        # use a NiN output for speed
        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n

def generator2DWide(t_image, scaleFactor, numResBlocks, numFilters, kernelSize, batchNorm=False, reuse=False):
    """ Generator in WDSR
    """
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
        shallow = n
        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64*6, (1, 1), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 48, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            n = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
        deep=n
        # upscale the shallow layer
        shallow = Conv2d(shallow, 3*4**2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        shallow = SubpixelConv2d(shallow, scale=4, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        # upscale the deep layer
        deep = Conv2d(deep, 3*4**2, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/1')
        deep = SubpixelConv2d(deep, scale=4, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        # add and output using nin
        n = ElementwiseLayer([deep, shallow], tf.add, name='add3')
        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n

# the TL discriminator, its more efficient
def discriminator2DOld(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        # to get started
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='h0/c')
        # expand features
        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
        
        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn')
        
        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn')
        # this section is too hardcore for a 1080ti
#        net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
#        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn')
#        
#        net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
#        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h5/bn')
#         compress features 1x1
#        net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
#        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
#        
#        net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
#        net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/bn')
        net_h7=net_h3
        net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn')
        # deep neighbouring features
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn2')
        
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='res/bn3')
        
        # skip connection
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)
#        net_h8=net
        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        #classify true false 
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/out')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits
    
    
# Ledigs original discriminator (quite heavy, and no skips)
def discriminator2D(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net = Conv2d(net_in, df_dim, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='h0/c')
        # upsample
        net = Conv2d(net, df_dim, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='h1/c')        
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
        numDiscrimBlocks=3
        for i in range(numDiscrimBlocks):
            expon=2**(i+1)
            # expand x2
            net = Conv2d(net, df_dim*expon, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name=f'db{i}/h1/c')        
            net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name=f'db{i}/h1/bn')
            # upsample
            net = Conv2d(net, df_dim*expon, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name=f'db{i}/h2/c')        
            net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name=f'db{i}/h2/bn')

        net_ho = FlattenLayer(net, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1024, act=lrelu, W_init=w_init, name='ho/dense')       
        #classify true false 
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/out')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits

#TODO: get this shit sorted like in cyclegan
def discrimLSGAN(input_images, is_train=True, reuse=False):
    
    output=input_images
    
    return output

def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        print("Building the VGG-19 network")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
#        assert red.get_shape().as_list()[1:] == [224, 224, 1]
#        assert green.get_shape().as_list()[1:] == [224, 224, 1]
#        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
#        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        conv = network
        """ fc 6~8 """
#        network = FlattenLayer(network, name='flatten')
#        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
#        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
#        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        return network, conv

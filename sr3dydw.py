'''
2D/3D super resolution routines
'''

import glob
from timeit import default_timer as timer
from sys import stdout
import pdb
import sys
import numpy as np
import datetime
import tensorflow as tf
import tensorlayer as tl
import h5py
from sr3dModels import *
from sr2dModels import *
import os
from skimage.measure import compare_psnr as psnr
import argparse
from PIL import Image
from tqdm import tqdm
from tifffile import imsave
#import png
from glcmLosses import *

# convert a 3D tensor into 2D slices
def convert22D(x):
    
    return x

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def summarise_model(layerVars):
    gParams=0
    for variable in layerVars:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        print(variable.name+f' numParams: {variable_parameters}')
        print(shape)
        gParams += variable_parameters
    print(f'Network Parameters: {gParams}')
    return gParams
    
def loadDataset(iterNum, batch_size, trainImageIDs, img_width, img_height, img_depth, scaleFactor, path, name, subset, downgrade, datasetBitDepth, numBits):
    
    numBatches=np.floor(np.max(trainImageIDs)/batch_size)
    index = np.mod(iterNum, numBatches)
    beg = index * batch_size
    end = (index + 1) * batch_size
    trainBatchIDs=trainImageIDs[int(beg):int(end)]
    if img_depth>1:
        hr_batch = np.zeros((len(trainBatchIDs), img_width, img_height, img_depth), dtype=datasetBitDepth)
        if scaleFactor=='M':
            lr_batch = np.zeros((len(trainBatchIDs), img_width, img_height, img_depth), dtype=datasetBitDepth)
        else:
            lr_batch = np.zeros((len(trainBatchIDs), img_width//scaleFactor, img_height//scaleFactor, img_depth//scaleFactor), dtype=datasetBitDepth)
    else:
        hr_batch = np.zeros((len(trainBatchIDs), img_width, img_height, 3), dtype=datasetBitDepth)
        if scaleFactor=='M':
            lr_batch = np.zeros((len(trainBatchIDs), img_width, img_height, 3), dtype=datasetBitDepth)
        else:
            lr_batch = np.zeros((len(trainBatchIDs), img_width//scaleFactor, img_height//scaleFactor, 3), dtype=datasetBitDepth)
    
    for i, id in enumerate(trainBatchIDs):
        hr_path = _hr_image_path(path, name, subset, id)
        lr_path = _lr_image_path(path, name, subset, downgrade, scaleFactor,id)
        lr = np.load(lr_path)
        hr = np.load(hr_path)
        # generate random origin for cropping
        if img_depth>1:
            if scaleFactor != 'M':
                # if shape is 80 (20 with scale 4), and specified size is 16, sample from 0 to 20-16/4 = 16
                lr_w = np.random.randint(lr.shape[0] - img_width//scaleFactor+1)
                lr_h = np.random.randint(lr.shape[1] - img_height//scaleFactor+1)
                lr_d = np.random.randint(lr.shape[2] - img_depth//scaleFactor+1)

                hr_w = lr_w * scaleFactor
                hr_h = lr_h * scaleFactor
                hr_d = lr_d * scaleFactor
                
                lr = lr[lr_w:lr_w + img_width//scaleFactor, lr_h:lr_h + img_height//scaleFactor, lr_d:lr_d + img_depth//scaleFactor]
                
                hr = hr[hr_w:hr_w + img_width, hr_h:hr_h + img_height, hr_d:hr_d + img_depth]
            else: # if multiscale, lr is hr, so crop equally
                lr_w = np.random.randint(lr.shape[0] - img_width+1)
                lr_h = np.random.randint(lr.shape[1] - img_height+1)
                lr_d = np.random.randint(lr.shape[2] - img_depth+1)
                lr = lr[lr_w:lr_w + img_width, lr_h:lr_h + img_height, lr_d:lr_d + img_depth]
                
                hr = hr[lr_w:lr_w + img_width, lr_h:lr_h + img_height, lr_d:lr_d + img_depth]
        else:
            if scaleFactor != 'M':
                # if shape is 80 (20 with scale 4), and specified size is 16, sample from 0 to 20-16/4 = 16
                lr_w = np.random.randint(lr.shape[0] - img_width//scaleFactor+1)
                lr_h = np.random.randint(lr.shape[1] - img_height//scaleFactor+1)

                hr_w = lr_w * scaleFactor
                hr_h = lr_h * scaleFactor
                
                lr = lr[lr_w:lr_w + img_width//scaleFactor, lr_h:lr_h + img_height//scaleFactor]
                
                hr = hr[hr_w:hr_w + img_width, hr_h:hr_h + img_height]
            else: # if multiscale, lr is hr, so crop equally
                lr_w = np.random.randint(lr.shape[0] - img_width+1)
                lr_h = np.random.randint(lr.shape[1] - img_height+1)

                lr = lr[lr_w:lr_w + img_width, lr_h:lr_h + img_height]
                
                hr = hr[lr_w:lr_w + img_width, lr_h:lr_h + img_height]
        
        lr_batch[i] = lr
        hr_batch[i] = hr
        
    lr_batch=(lr_batch-numBits)/numBits #python auto casts this to float64
    hr_batch=(hr_batch-numBits)/numBits
    if img_depth>1:
        lr_batch=np.expand_dims(lr_batch, 4)
        hr_batch=np.expand_dims(hr_batch, 4)
    return lr_batch, hr_batch


def _hr_image_path(path, name, subset, id):
    return os.path.join(path, f'{name}_{subset}_HR', f'{id:04}.npy') #TODO the 4 trailing zeros is legacy and I CBF removing it because itll fuck up all the other datasets

def _lr_image_path(path, name, subset, downgrade, scale, id):
    return os.path.join(path, f'{name}_{subset}_LR_{downgrade}_X{scale}', f'{id:04}x{scale}.npy')

def int_range(s):
    try:
        fr, to = s.split('-')
        return range(int(fr), int(to) + 1)
    except Exception:
        raise argparse.ArgumentTypeError(f'invalid integer range: {s}')
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def str2int(v):
    if v=='M':
        return v
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError('int value expected.')
    return v
    
def str2float(v):
    if v=='M':
        return v
    try:
        v = float(v)
    except:
        raise argparse.ArgumentTypeError('float value expected.')
    return v
        
##
# Module Flags
##
def argumentParser():
    parser = argparse.ArgumentParser(description='SR/GAN3D')
    parser.add_argument('--preprocess', type=str2bool, default=False, help='Flag to activate if mat volumes need to be numpy-ified')

    parser.add_argument('--train', type=str2bool, default=False, help='Flag to activate for training')

    parser.add_argument('--gan', type=str2bool, default=False, help='Flag to activate for GAN modelling')

    parser.add_argument('--test', type=str2bool, default=False, help='Flag to activate for testing')

    parser.add_argument('--cpu', type=str2bool, default=False, help='Flag to activate for testing')

    parser.add_argument('--predict', type=str2bool, default=False, help='Flag to activate for predicting')

    ##
    # Conversion Arguments
    ##

    parser.add_argument('--indir', type=str, default='./DRSRD3/DRSRD3_3D/shuffled3D', help='path to DIV2K images')

    parser.add_argument('--outdir', type=str, default='./shuffled3D_BIN', help='directory where converted image files are stored')

    parser.add_argument('--inext', type=str, default='png', help='Input image file type')
    
    parser.add_argument('--inBits', type=str, default='uint8', help='Input image bit depth')

    ##
    # Model Parameters
    ##

    parser.add_argument('--residual_blocks', default=16, help='Number of residual blocks')

    parser.add_argument('--numFilters', default=64, help='Number of filters')

    parser.add_argument('--batchNorm', type=str2bool, default=False, help='batchNorm Layers')

    parser.add_argument('--activation', type=str, default='prelu', help='activation Layers')

    parser.add_argument('--scaleFactor', type=str2int, default='4', help='Upsampling factor')

    parser.add_argument('--downgrade', default='unknown', help='Upsampling method')

    parser.add_argument('--multiScale', type=str2bool, default=False, help='Use the multiscale Model')

    parser.add_argument('--gLoss', type=str, default='L1', help='L1, L2, or perceptual loss for generator')

    parser.add_argument('--perceptual', type=str2bool, default=False, help='perceptual loss for generator')
    
    parser.add_argument('--glcmRatio', type=str2float, default=0, help='GLCM loss coefficient')

    parser.add_argument('--gRatio', type=str2float, default=1e-5, help='perceptual loss coefficient')

    parser.add_argument('--gdRatio', type=str2float, default=0, help='6point Gradient loss coefficient')

    parser.add_argument('--advRatio', type=str2float, default=1e-3, help='Adversarial and Discriminatory Coefficient')

    parser.add_argument('--featureRatio', type=str2float, default=0.5, help='target vgg loss ratio')

    parser.add_argument('--classificationRatio', type=str2float, default=0.1, help='target discriminatory ratio') # if this is negative, tune against Dloss = 1

    parser.add_argument('--hyperFlag', type=str2bool, default=False, help='Dynamic scaling of the Adversarial and Discriminatory Coefficients')

    parser.add_argument('--hyperBCFlag', type=str2bool, default=False, help='Bicubic scaling of the Adversarial and Discriminatory Coefficient')


    parser.add_argument('--batch-size',type=str2int, default=16)
    parser.add_argument('--width', type=str2int, default=64)
    parser.add_argument('--height', type=str2int, default=64)
    parser.add_argument('--depth', type=str2int, default=64)
    parser.add_argument('--num-epochs', type=str2int, default=250)
    parser.add_argument('--ganEpoch', type=str2int, default=100)
    parser.add_argument('--iterMax',type=str2int, default=1000)
    parser.add_argument('--learnRate',type=str2float, default=1e-4) #pump up the LR to compensate for TF bad init
    parser.add_argument('--decayRate',type=str2float, default=0.1) #pump up the LR to compensate for TF bad init
    parser.add_argument('--decaySteps',type=str2float, default=50) #pump up the LR to compensate for TF bad init

    parser.add_argument('--valW', type=str2int, default=500)
    parser.add_argument('--valH', type=str2int, default=500)
    parser.add_argument('--valD', type=str2int, default=1)
    # Model IO Arguments
    
    parser.add_argument('--diskFlag',type=str2bool, default=True, help='dataset loading')
    
    parser.add_argument('--dataset',type=str, default='./shuffled3D_BIN', help='path to dataset')
    
    parser.add_argument('--datasetBits', type=str, default='uint8', help='Dataset bit depth')
   
    parser.add_argument('--trainIDs', type=int_range, default='1-2400', help='training image ids')

    parser.add_argument('--valIDs', type=int_range, default='2401-2700', help='validation image ids')
        
    parser.add_argument('--path_prediction',type=str, default='./output', help='Path to save training predictions')

    parser.add_argument('--path_volumes',type=str, default='./output', help='Path to save test volumes')

    parser.add_argument('--checkpoint_dir',type=str, default='./output', help='Path to save checkpoints')

    parser.add_argument('--restore', default=None, help='Checkpoint path to restore training')

    parser.add_argument('--testDir', default=None, help='test path')

    parser.add_argument('--contEpoch', default=0, help='Restart epoch')

    parser.add_argument('--valVisInterval', type=str2int, default=10, help='visualisation during training and validation')

    args = parser.parse_args()
    
    return args
'''
from gooey import Gooey

@Gooey'''
def main():
    args=argumentParser()
    
    convertFlag=args.preprocess
    trainFlag=args.train
    ganFlag=args.gan
    testFlag=args.test
    glcmRatio = args.glcmRatio
    gLoss=args.gLoss
    gdRatio=args.gdRatio
    hyperFlag=args.hyperFlag
    hyperBCFlag=args.hyperBCFlag
    iterMax=args.iterMax
    cpuFlag=args.cpu
    batchNorm=args.batchNorm
    activation=args.activation
    valVisInterval=args.valVisInterval
    diskFlag=args.diskFlag
    preProcessingBitDepth=args.inBits
    
    datasetBitDepth=args.datasetBits

    '''

    IMAGE CONVERSION ROUTINE

    '''
    #TODO: switch to matplotlib instead of PIL
    if convertFlag:
        print('Running image conversion on input images')

        ''' start converter '''

        input_path=args.indir
        output_path=args.outdir
        extension=args.inext

        # generate the read paths
        img_paths=[]
        img_paths_ext = glob.glob(os.path.join(input_path, '**', f'*.{extension}'), recursive=False)
        img_paths.extend(img_paths_ext)

        # convert and save
        for img_path in tqdm(img_paths):
            img_dir, img_file = os.path.split(img_path)
            img_id, img_ext = os.path.splitext(img_file)

            rel_dir = os.path.relpath(img_dir, input_path)
            out_dir = os.path.join(output_path, rel_dir)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if extension=='png':
                if preProcessingBitDepth == 'uint8':
                    img = Image.open(img_path)
                    if img.mode != 'RGB': #makes it triple channel
                        img = img.convert('RGB')
                    img = np.array(img, dtype=datasetBitDepth)
                elif preProcessingBitDepth == 'uint16':
                    reader = png.Reader(img_path) #input must be already triple channel, otherwise will save as single channel 16bit
                    data = reader.asDirect()
                    pixels = data[2]
                    img = []
                    for row in pixels:
                        row = np.asarray(row)
                        row = np.reshape(row, [-1, 3])
                        img.append(row)
                    img = np.stack(img, 1)
                    img = np.rot90(img,-1)
                    img = np.fliplr(img)

            elif extension=='mat':
                #img=octave.load(img_path)
                #img=img.temp
#                img=sio.loadmat(img_path)
#                img=img['temp']
                arrays = {}
                f = h5py.File(img_path)
                for k, v in f.items():
                    arrays[k] = np.array(v)
                img=arrays['temp']
            arr_path = os.path.join(out_dir, f'{img_id}.npy')
            np.save(arr_path, np.array(img, dtype=preProcessingBitDepth))
            
        print('Done')
    '''

    TRAINING AND TESTING

    '''
    #TODO: make vgg compatible with 3d on a slice basis
    if trainFlag or testFlag:
        # argument parsing
    #    if testFlag:
    #        ganFlag=False
        scaleFactor=args.scaleFactor
        numFilters=int(args.numFilters)

        residual_blocks=int(args.residual_blocks)
        perceptFlag=args.perceptual
        path_prediction=args.path_prediction
        checkpoint_dir=args.checkpoint_dir
        if datasetBitDepth == 'uint8':
            numBits=127.5
        elif datasetBitDepth == 'uint16':
            numBits=32767.5
        img_width=args.width
        img_height=args.height
        img_depth=args.depth # if the depth given is 1, set the 2d flag on
        valW=args.valW
        valH=args.valH
        valD=args.valD
        if img_depth == 1:
            flatFlag=True
            dim='2D'
        else:
            flatFlag=False
            dim='3D'
        batch_size=args.batch_size
        restore=args.restore
        trainImageIDs=args.trainIDs
        valImageIDs=args.valIDs
        epochs=args.num_epochs
        ganEpoch=args.ganEpoch
        path=args.dataset
        iterations_train=iterMax
        learnRate=args.learnRate
        decay_rate = args.decayRate
        decay_steps = args.decaySteps #drop 1 order of magnitude in xxx epochs by exponential
        if ganFlag:
            decay_steps=decay_steps*2
        advRatio=args.advRatio
        gRatio=args.gRatio
        featureRatio=args.featureRatio
        classificationRatio=args.classificationRatio
        eps=1e-12
        
        # define the models
        
        # create the learning rate variable
        with tf.variable_scope('gRatio'):
            gRatio = tf.Variable(gRatio, trainable=False)
        with tf.variable_scope('advRatio'):
            advRatio = tf.Variable(advRatio, trainable=False)
        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(learnRate, trainable=False)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = lr_v#tf.train.exponential_decay(lr_v, global_step=global_step, decay_rate=decay_rate, decay_steps=decay_steps) # learning rate decays by 
        if ganFlag: #gan can only discriminate on set shapes
            if flatFlag:
                outputShape=[batch_size, img_width, img_height, 3]
            else:
                outputShape=[batch_size, img_width, img_height, img_depth, 1]
        else: # if gan is off, model can be applied to any size
            if flatFlag:
                outputShape=[None, None, None, 3]
            else:
                outputShape=[None, None, None, None, 1]
        # the ground truth image tensor
        t_target_image = tf.placeholder('float32',outputShape, name='t_HR_target_image')
        
        if flatFlag:# the generator is always fully convolutional
            inputShape=[None, None, None, 3]
        else:
            inputShape=[None, None, None, None, 1]
        inputTensor = tf.placeholder('float32',inputShape, name='t_LR_image_input_to_SR_generator')
        #TODO: add option for argument to handle different models
        # pass to the generator model
        if args.multiScale: #3D multiscale SRCNN model (deprecated?)
            net_gen = generatorMS(input_gen=inputTensor, kernel=3, nb=residual_blocks, numFilters=numFilters, is_train=True, reuse=False)
        elif flatFlag: # 2D EDSR model (built on TL)
            net_gen = generator2D(inputTensor, scaleFactor, residual_blocks, numFilters, 3, initKernelFactor=1, activation=activation, batchNorm=batchNorm, reuse=False)
            net_gen_data=net_gen
            net_gen=net_gen_data.outputs
        else: #3d models
            net_gen = generatorTF(inputTensor, 3, residual_blocks, 64, scaleFactor, False, True)
            #net_gen = generator(input_gen=inputTensor, kernel=3, nb=residual_blocks, scaleFactor=scaleFactor, numFilters=numFilters, reuse=False)
        # define the basic mse loss
        if gLoss=='L1':
            mse_loss = tf.reduce_mean(tf.abs(net_gen - t_target_image), name='MSEGeneratorLoss')
        elif gLoss=='L2':
            mse_loss = tf.reduce_mean(tf.square(net_gen - t_target_image), name='MSEGeneratorLoss')
        elif gLoss=='None':
            mse_loss = tf.zeros(1,tf.float32)
        g_loss = mse_loss
        g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
        g_psnr = tf.image.psnr(tf.squeeze(net_gen), tf.squeeze(t_target_image), max_val=2)
        
        if glcmRatio>0:
            print('GLCM Loss function is active')
            numLevels = 8
            span = 2#scaleFactor
            glcmLoss = glcmRatio*tf.reduce_mean(tf.abs(compute8WayGLCM(net_gen, numLevels, span) - compute8WayGLCM(t_target_image, numLevels, span)), name = 'GLCMGeneratorLoss')
        else:
            glcmLoss = tf.zeros(1,tf.float32)
        # calculate gradients to calculate the gradent discrepancy (guides the GAN)
        g_loss = g_loss + glcmLoss
        if gdRatio>0:
            print('6-point Gradient Loss function is active')
            if flatFlag: # channels might cause issues here?
                dx_real = t_target_image[:, 1:, :, :] - t_target_image[:, :-1, :, :]
                dy_real = t_target_image[:, :, 1:, :] - t_target_image[:, :, :-1, :]

                dx_fake = net_gen[:, 1:, :, :] - net_gen[:, :-1, :, :]
                dy_fake = net_gen[:, :, 1:, :] - net_gen[:, :, :-1, :]

                gd_loss = gdRatio*(tf.reduce_mean(tf.square(tf.abs(dx_real) - tf.abs(dx_fake))) + tf.reduce_mean(tf.square(tf.abs(dy_real) - tf.abs(dy_fake))))
            else:
                dx_real = t_target_image[:, 1:, :, :, :] - t_target_image[:, :-1, :, :, :]
                dy_real = t_target_image[:, :, 1:, :, :] - t_target_image[:, :, :-1, :, :]
                dz_real = t_target_image[:, :, :, 1:, :] - t_target_image[:, :, :, :-1, :]
                
                dx_fake = net_gen[:, 1:, :, :, :] - net_gen[:, :-1, :, :, :]
                dy_fake = net_gen[:, :, 1:, :, :] - net_gen[:, :, :-1, :, :]
                dz_fake = net_gen[:, :, :, 1:, :] - net_gen[:, :, :, :-1, :]

                gd_loss = gdRatio*(tf.reduce_mean(tf.square(tf.abs(dx_real) - tf.abs(dx_fake))) + tf.reduce_mean(tf.square(tf.abs(dy_real) - tf.abs(dy_fake))) + tf.reduce_mean(tf.square(tf.abs(dz_real) - tf.abs(dz_fake))))
        else:
            gd_loss=tf.zeros(1,tf.float32)
        g_loss = g_loss + gd_loss
        # define the vgg loss
        if perceptFlag and trainFlag:
            print('VGG Loss function is active')
            if flatFlag:
                vggHRInput=t_target_image
                vggSRInput=net_gen
            else:
                vggHRInput=convert22D(t_target_image)
                vggSRInput=convert22D(net_gen)
            ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
            t_target_image_224 = vggHRInput#tf.image.resize_images(vggHRInput, size=[224, 224], method=0, align_corners=False)
            
            t_predict_image_224 = vggSRInput#tf.image.resize_images(vggSRInput, size=[224, 224], method=0, align_corners=False)

            net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
            _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

            # vgg ouputs are always 2D
            vgg_loss = gRatio*tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    #        _, _, vggHRInput = discriminator2DOld(t_target_image, is_train=False, reuse=True)
    #        _, _, vggSRInput = discriminator2DOld(net_gen, is_train=False, reuse=True)  
    #        vgg_loss = gRatio*tl.cost.mean_squared_error(vggHRInput, vggSRInput, is_mean=True)
        else:
            vgg_loss = tf.zeros(1,tf.float32)
            
        g_loss = g_loss + vgg_loss
        ganStr=''
        if ganFlag and trainFlag:
            print('ADV Loss function is active - delayed activation - check the ganEpoch')
            ganStr='-gan'
            net_gen.set_shape(outputShape)
            #create discriminators (this is just one discriminator taking in a dataset split into 2 parts)
            if flatFlag:
                disc_out_real, logits_real = discriminator2D(t_target_image, is_train=True, reuse=False)
                disc_out_fake, logits_fake = discriminator2D(net_gen, is_train=True, reuse=True)  
                disc_out_real_data=disc_out_real
                disc_out_real=disc_out_real_data.outputs    
                disc_out_fake_data=disc_out_fake
                disc_out_fake=disc_out_fake_data.outputs    
                            
            else:
                disc_out_real, logits_real = discriminatorTF(input_disc=t_target_image, kernel=3, is_train=True, reuse=False)
                disc_out_fake, logits_fake = discriminatorTF(input_disc=net_gen, kernel=3, is_train=True, reuse=True)

            with tf.variable_scope('Discriminator_loss'):
                # the cross entropy should approach zero for perfect discrimination. random discrimination should be 0.5 for both, summing to 1           
    #            d_loss = (d_loss_real + d_loss_fake)
                
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
                
                d_loss = d_loss1 + d_loss2
            # here, fake is 0 if the discriminator is good.
            # the generator aims to raise the label values towards 1
            with tf.variable_scope('Adversarial_loss'):
                g_gan_loss = advRatio*tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')#tf.reduce_mean(-tf.log(disc_out_fake+eps))
            
            g_loss = g_gan_loss + g_loss

            d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)
    #        dsaver = tf.train.Saver(d_vars, max_to_keep = 10000)
            # compile the discriminator model
            d_optim = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars, global_step=global_step)
            # compile the mse generator
            gMSE_optim = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss, var_list=g_vars, global_step=global_step)
        
        # compile the overall generator model
        g_optim = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars, global_step=global_step)

        # Resources
        if cpuFlag:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1
        session = tf.Session(config=config)
    #    tl.layers.initialize_global_variables(session)
        session.run(tf.initialize_all_variables())
        
        
        t_vars = tf.compat.v1.trainable_variables()
        total_parameters = summarise_model(t_vars)
        print(f'Total Network Parameters: {total_parameters}')
    
    #    tf.global_variables_initializer()
        # batch norm layers in generator are ignorable this way, so edsr can slot into resnet
        # define the vgg loss
        if perceptFlag and trainFlag:
                # load in the vgg weights
            vgg19_npy_path = "vgg19.npy"
            npz = np.load(vgg19_npy_path, encoding='latin1').item()
            del npz['fc6']
            del npz['fc7']
            del npz['fc8']
            params = []
            for val in sorted(npz.items()):
                W = np.asarray(val[1][0])
                b = np.asarray(val[1][1])
                print("Loading VGG-19 layer %s: %s, %s" % (val[0], W.shape, b.shape))
                params.extend([W, b])
            tl.files.assign_params(session, params, net_vgg)
    #    # temporary TL combatibility weights
    #    tl.global_flag['mode'] = 'srgan'
    #    checkpoint_dir='/home/user/machineLearning/srgan/checkpoint'
    #    if tl.files.load_and_assign_npz(sess=session, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_gen_data) is False:
    #        tl.files.load_and_assign_npz(sess=session, name=checkpoint_dir disc_out_real+ '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_gen)
    #    tl.files.load_and_assign_npz(sess=session, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=disc_out_real_data)
        
        # load weights
        # TODO: either migrate sr3d back to TL, or write own npz saver/loader for tf
        # pdb.set_trace()
        #if not flatFlag:
        saver = tf.train.Saver(g_vars, max_to_keep = 10000)

        if restore is not None:
            val_restore = [int(s) for s in restore.split('-') if s.isdigit()]
            val_restore=val_restore[-1]
            if not flatFlag:
                saver.restore(session, restore)
            else:
                saver.restore(session, restore)
    #            restore=sorted(glob.glob(restore+'*'))
    #            if ganFlag:
    #                load_params = tl.files.load_npz(name=restore[0])
    #                tl.files.assign_params(session, load_params, disc_out_real_data)
    #                load_params = tl.files.load_npz(name=restore[1])
    #                tl.files.assign_params(session, load_params, net_gen_data)
    #            else:
    #                load_params = tl.files.load_npz(name=restore[1])
    #                tl.files.assign_params(session, load_params, net_gen_data)
    #            val_restore = restore[0].split('/')[-1]
    #            val_restore=int(val_restore[6:10]) #the fuck is this
            session.run(tf.assign(learning_rate, learnRate))

            #g_vars
            #gImport = tf.train.list_variables(restore)
        else:
            val_restore = 0

        array_psnr = []
        array_ssim = []
        
        downgrade=args.downgrade
        name = path.split('_')[0] # this is the non BIN
        name = name.split('/')[-1]
        # training routine
        
        if trainFlag:
            checkPath=path.split('/')[-1]
            rightNow=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            trainingDir=f"{checkpoint_dir}/{rightNow}-SR{dim}{ganStr}-{activation}-x{scaleFactor}-{checkPath}-numBlocks-{residual_blocks}-numFilters-{numFilters}-loss-{gLoss}"
            os.mkdir(trainingDir)
            # save the input arguments for recordkeeping
            with open(os.path.join(trainingDir, 'args.txt'), 'w') as f:
                for k, v in sorted(args.__dict__.items()):
                    f.write(f'{k}={v}\n')
            
            trainOutputDir=f'./trainingOutputs/{rightNow}-SRXD{dim}{ganStr}-{activation}-scale-{scaleFactor}-{checkPath}-numBlocks-{residual_blocks}-numFilters-{numFilters}-loss-{gLoss}'
            os.mkdir(trainOutputDir)
            
            # load the entire dataset into the RAM if it fits
            if diskFlag is not True:
                print('Loading entire dataset into RAM')
                trainLR, trainHR = loadDataset(0, len(trainImageIDs), trainImageIDs, img_width, img_height, img_depth, scaleFactor, path, name, 'train', downgrade, datasetBitDepth, numBits)
                
                valLR, valHR = loadDataset(0, len(valImageIDs), valImageIDs, valW, valH, valD, scaleFactor, path, name, 'valid', downgrade, datasetBitDepth, numBits)
            #redefine this for hyperdynamictuning

            meanValPSNR=0
            oldMaxValPSNR=0
            metrics=np.zeros(epochs + val_restore)
            for epochNum in range(val_restore, epochs + val_restore):
                start=timer()
                #TODO: change the LR adaptation by setting LR as None and passing a value during TF.run
                if oldMaxValPSNR>meanValPSNR*1.5:
                    learnRate = learnRate/2
                learnRateNew=learnRate*decay_rate**((epochNum-val_restore)/decay_steps)

                session.run(tf.assign(learning_rate, learnRateNew))
                # arrays for storing iteration averaged metrics
                trainingPSNR=np.zeros(iterations_train)
                trainingMSE=np.zeros(iterations_train)
                trainingGLCM=np.zeros(iterations_train)
                trainingVGG=np.zeros(iterations_train)
                trainingADV=np.zeros(iterations_train)
                trainingDLOSS=np.zeros(iterations_train)
                for iterNum in range(iterations_train):
                    #load training batch
                    if diskFlag:
                        lr_train, hr_train = loadDataset(iterNum, batch_size, trainImageIDs, img_width, img_height, img_depth, scaleFactor, path, name, 'train', downgrade, datasetBitDepth, numBits)
                    else:
                        inds=(np.arange(batch_size)+1)+iterNum*batch_size
                        lr_train = trainLR[inds,:,:,:] 
                        hr_train = trainHR[inds,:,:,:]
                    #TODO: there are several obvious logical flows that are neater than this.
                    # run SR/GAN
                    if ganFlag:
                        if epochNum < ganEpoch: # run mse and glcm only, no vgg or gan
                            # update G
                            errglcm, errmse, tpsnr, learn_Rate,_ = session.run([glcmLoss, mse_loss, g_psnr, learning_rate, gMSE_optim], {inputTensor: lr_train, t_target_image: hr_train})    
                            batchPSNR=np.mean(tpsnr)
                            stdout.write("\rLR: {%.4e} Epoch [%4d/%4d] [%4d/%4d]: glcmloss: %.4e (mse: %.4e) [psnr: %.4f]" % (learn_Rate, epochNum+1, epochs + val_restore, iterNum+1, iterations_train, errglcm, errmse, np.mean(tpsnr)))
                        else:
                            # update G and generate fakes
                            errg, errglcm, errmse, errgan, errgd, errvgg, tpsnr, learn_Rate, alpha, beta, _ = session.run([g_loss, glcmLoss, mse_loss, g_gan_loss, gd_loss, vgg_loss,  g_psnr, learning_rate, gRatio, advRatio, g_optim], {inputTensor: lr_train, t_target_image: hr_train})

                            # update D by feeding it reals and fakes
                            errd, errdR, errdF, _ = session.run([d_loss, disc_out_real, disc_out_fake, d_optim], {t_target_image: hr_train, inputTensor: lr_train})
                            batchPSNR=np.mean(tpsnr)
                            stdout.write("\rLR: {%.4e} Epoch [%4d/%4d] [%4d/%4d]: d_loss: %.4f dR: %.4f dF: %.4f g_loss: %.4e (mse: %.4e glcm: %.4e gdl: %.4e vgg: %.4e adv: %.4e) [psnr: %.4f]" % (learn_Rate, epochNum+1, epochs + val_restore, iterNum+1, iterations_train, errd, np.mean(errdR), np.mean(errdF), errg, errmse, errglcm, errgd, errvgg, errgan, np.mean(tpsnr)))
                            trainingVGG[iterNum]=errvgg
                            trainingADV[iterNum]=errgan
                            trainingDLOSS[iterNum]=errd
                    else: #run SRXD 
                        # update G
                        errglcm, errmse, tpsnr, learn_Rate, _ = session.run([glcmLoss, mse_loss, g_psnr, learning_rate, g_optim], {inputTensor: lr_train, t_target_image: hr_train})    
                        batchPSNR=np.mean(tpsnr)
                        stdout.write("\rLR: {%.4e} Epoch [%4d/%4d] [%4d/%4d]: glcmLoss: %.4e (mse: %.4e) [psnr: %.4f]" % (learn_Rate, epochNum+1, epochs + val_restore, iterNum+1, iterations_train, errglcm, errmse, batchPSNR))
                    stdout.flush()
                    
                    trainingPSNR[iterNum]=batchPSNR
                    trainingMSE[iterNum]=errmse
                    trainingGLCM[iterNum]=errglcm
                    
                stdout.write("\n")
                print('Mean metrics: MSE: %.4e, GLCM: %.4e, PSNR: %.4f, VGG: %.4e, ADV: %.4e, DLOSS: %.4f' %(np.mean(trainingMSE), np.mean(trainingGLCM), np.mean(trainingPSNR), np.mean(trainingVGG), np.mean(trainingADV), np.mean(trainingDLOSS)))
                
                if ganFlag and epochNum >= ganEpoch and hyperFlag:
                    meanMSE = np.mean(trainingMSE)
                    meanVGG = np.mean(trainingVGG)
                    meanADV = np.mean(trainingADV)
                    print('Hyperparameter Tuning Module is active, scaling alpha and beta by ratio method')
                    # target all losses to have some ratios
                    # adjust the ratios bases on the current misalignment of target ratios
                    alpha=alpha*meanMSE/meanVGG*featureRatio
                    if classificationRatio>0:
                        beta=beta*meanMSE/meanADV*classificationRatio
                    else: # tune against dloss = 1. beta is high is dloss is less than 1 and vice versa 
                        beta=beta/np.mean(trainingDLOSS)
                    print('Adjusting Loss Weights: alpha: %.4e, beta: %.4e' %(alpha, beta))
                    session.run(tf.assign(gRatio, alpha))
                    session.run(tf.assign(advRatio, beta))
                # load validation batch after epoch done
                #print('Validating End-of-Epoch Model')
                # load the validation dataset
                numValIterations=len(valImageIDs)
                valid_psnr=np.zeros(numValIterations)
                #valid_ssim=np.zeros(numValIterations)
                if np.mod(epochNum+1, valVisInterval)==0 or epochNum==val_restore or epochNum==val_restore + epochs:
                    os.mkdir(f'{trainOutputDir}/epoch-{(epochNum+1):04}')
                    if hyperBCFlag and epochNum > ganEpoch: #start tuning if gan has started
                        print('Hyperparameter Tuning Module is active, scaling alpha and beta by secant method')
                        meanDLoss=np.mean(trainingDLOSS)
                        if epochNum +1 <= val_restore + ganEpoch + valVisInterval:# if gan training just stabilised, initialise
                            oldMeanDLoss=meanDLoss
                            oldAlpha=alpha
                            oldBeta=beta
                            if meanDLoss>1:
                                alpha=alpha*0.1
                                beta=beta*0.1
                            else:
                                alpha=alpha*10
                                beta=beta*10
                            print('Adjusting Loss Weights: alpha: %.4e, beta: %.4e' %(alpha, beta))
                            session.run(tf.assign(gRatio, alpha))
                            session.run(tf.assign(advRatio, beta))
                        else:
                            # get the line equation and make the next guess at the intercept
                            # fetch gradients
                            gradientA=(meanDLoss-oldMeanDLoss)/(alpha-oldAlpha)
                            gradientB=(meanDLoss-oldMeanDLoss)/(beta-oldBeta)
                            # pass back values
                            oldMeanDLoss=meanDLoss
                            oldAlpha=alpha
                            oldBeta=beta
                            # compute new values
                            alpha=(1-meanDLoss+gradientA*alpha)/gradientA
                            beta=(1-meanDLoss+gradientB*beta)/gradientB
                            
                            if alpha<0:
                                if meanDLoss>1:
                                    alpha=oldAlpha*0.1
                                else:
                                    alpha=oldAlpha*10
                            if beta<0:
                                if meanDLoss>1:
                                    beta=oldBeta*0.1
                                else:
                                    beta=oldBeta*10
                            
                            
                            print('Adjusting Loss Weights: alpha: %.4e, beta: %.4e' %(alpha, beta))
                            session.run(tf.assign(gRatio, alpha))
                            session.run(tf.assign(advRatio, beta))

                for n in range(numValIterations):
                    if diskFlag:
                        lr_val, hr_val = loadDataset(n, 1, valImageIDs, valW, valH, valD, scaleFactor, path, name, 'valid', downgrade, datasetBitDepth, numBits)
                    else:
                        inds=(np.arange(1)+1)+n*1
                        lr_val = valLR[inds,:,:,:] 
                        hr_val = valHR[inds,:,:,:]
    #                totalpsnr = 0
    #                totalssim = 0
    #                array_psnr = np.empty(epochs)
                    #array_ssim = np.empty(epochs)
                    
                    hiRes = hr_val
                    superRes = session.run(net_gen, {inputTensor: lr_val})
    #                hiRes=(hiRes+1)*numBits
    #                
    #                superRes=superRes-(superRes.min())
    #                superRes = superRes/superRes.max()*numBits
                    val_max = np.max((np.amax(superRes), np.amax(hiRes)))
                    val_min = np.min((np.amin(superRes), np.amin(hiRes)))
                    SNR=2#(val_max-val_min) 
                    
                    valid_psnr[n] = psnr(hiRes, superRes, data_range = SNR)
                    #valid_ssim[n] = ssim(hiRes, superRes, data_range = SNR, multichannel=True)
                    
                    #hiRes=(hiRes+1)*numBits
                    superRes=(superRes+1)*numBits
    #                if flatFlag:
    #                    temp=np.concatenate((hiRes[0], superRes[0]), axis=1)
    #                else:
    #                    temp=np.concatenate((hiRes[0,0], superRes[0,0]), axis=1)
    #                scipy.misc.toimage(np.squeeze(temp.astype('int')), cmin=0, cmax=numBits).save(f'./trainingOutputs/{rightNow}-SR3D{ganStr}-scale-{scaleFactor}-numBlocks-{residual_blocks}-numFilters-{numFilters}/epoch-{epochNum}-{n+1:04}.png')
                    if np.mod(epochNum+1, valVisInterval)==0 or epochNum==val_restore or epochNum==val_restore + epochs:
                        imsave(f'{trainOutputDir}/epoch-{(epochNum+1):04}/{n+1:04}.tif', np.array(np.squeeze(superRes.astype('int')), dtype=datasetBitDepth))

                    stdout.write("\rValidation: [%4d/%4d] [PSNR: %.4f]" % (n+1, numValIterations, valid_psnr[n]))
                    stdout.flush()
                stdout.write("\n")
                end=timer()
                meanValPSNR=np.mean(valid_psnr)
                print('Mean Val PSNR: %4f Epoch Time: %4f' %(meanValPSNR, end-start))
                if meanValPSNR>oldMaxValPSNR:
                    maxValPSNR=meanValPSNR
                
                if np.mod(epochNum+1, valVisInterval)==0 or epochNum==val_restore or epochNum==val_restore + epochs or meanValPSNR>oldMaxValPSNR:
                    if not flatFlag:
                        save_path = saver.save(session, f"{trainingDir}/epoch-{epochNum+1}-PSNR-{meanValPSNR}.ckpt")
                    else:
                        save_path = saver.save(session, f"{trainingDir}/epoch-{epochNum+1}-PSNR-{meanValPSNR}.ckpt")
    #                    tl.files.save_npz(net_gen_data.all_params, name=f"{trainingDir}/epoch-{(epochNum+1):04}-PSNR-{meanValPSNR}-generator.npz", sess=session)
    #                    if ganFlag:
    #                        tl.files.save_npz(disc_out_real_data.all_params, name=f"{trainingDir}/epoch-{(epochNum+1):04}-PSNR-{meanValPSNR}-discriminator.npz", sess=session)

                
                oldMaxValPSNR=maxValPSNR
                metrics[epochNum]=meanValPSNR
    #            x=np.linspace(0,epochNum+1)
    #            y=metrics
    #            plotTerminal(x, y)
            # save the mean metrics as a txt file 
            np.savetxt(f'{trainOutputDir}/trainingLog.txt',metrics)
            
            
                # save generator and discriminator model as .ckpt
                
        if testFlag: # piggy back of the variables
            if flatFlag:    
                outdir=args.testDir+'/srOutputs-'+str(val_restore)
                os.makedirs(outdir, exist_ok=True)
                pngs = glob.glob(os.path.join(args.testDir, '*.png'), recursive=True)
                for path in pngs:
                    print('Super-resolve image ', path)
                    #pdb.set_trace()
                    if datasetBitDepth == 'uint8':
                        lr = Image.open(path)
                        if lr.mode != 'RGB':
                            lr = lr.convert('RGB')
                        lr = np.array(lr, dtype=datasetBitDepth)
                    elif datasetBitDepth == 'uint16':
                        reader = png.Reader(path)
                        data = reader.asDirect()
                        pixels = data[2]
                        lr = []
                        for row in pixels:
                            row = np.asarray(row)
                            row = np.reshape(row, [-1, 3])
                            lr.append(row)
                        lr = np.stack(lr, 1)
                        lr=np.rot90(lr,-1)
                        lr=np.fliplr(lr)
                        
                    lr_batch = np.zeros((1, lr.shape[0], lr.shape[1], 3), dtype=datasetBitDepth)
                    lr_batch[0] = lr
                    lr=lr_batch
                    #lr = lr[:,0:100, 0:100, :]
                    lr = (lr-numBits)/numBits
                    sr = session.run(net_gen, {inputTensor: lr})
                    sr = (sr+1)*numBits
                    #sr = sr.astype(datasetBitDepth)
                    #sr = np.array(np.squeeze(sr.astype('int')))
                    head, tail = os.path.split(path)
                    name, ext = os.path.splitext(tail)
                    imsave(f'{outdir}/{name}-sr{ganStr}.tif', np.array(np.squeeze(sr.astype('int')), dtype=datasetBitDepth))

                    # parse all checkpoints
                    
                    # for all checkpoints
                    
                    # load checkpoints
                    #saver.restore(session, restore)
            else:
                outdir=args.testDir+'/srOutputs-'+str(val_restore)
                os.makedirs(outdir, exist_ok=True)
                pngs = glob.glob(os.path.join(args.testDir, '*.mat'), recursive=True)
                for path in pngs:
                # load HR LR batches
                    print('Super-resolve image ', path)
                    arrays = {}
                    f = h5py.File(path)
                    for k, v in f.items():
                        arrays[k] = np.array(v)
                    img=arrays['temp']
                    img=np.expand_dims(img, 0)
                    img=np.expand_dims(img, 4)
                    img=(img-numBits)/numBits
                    #img=img[:,0:100, 0:100, 0:100, :]
                    # predict values
                    #pdb.sdfa
                    superRes = session.run(net_gen, {inputTensor: img})
                    superRes=(superRes+1)*numBits
                    # compute metrics
                    head, tail = os.path.split(path)
                    name, ext = os.path.splitext(tail)
                    # save metrics and SR images
                    imsave(f'{outdir}/{name}-SR3D.tif', np.array(np.squeeze(superRes.astype('int')), dtype=datasetBitDepth))

if __name__ == '__main__':
    main()













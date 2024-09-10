from __future__ import division
import os, scipy.io
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# import tensorflow.contrib.slim as slim
import tf_slim as slim
import numpy as np
import rawpy
import glob
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd



input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
# input_dir = '../Learning-to-See-in-the-Dark/dataset/Sony/short/'
# gt_dir = '../Learning-to-See-in-the-Dark/dataset/Sony/long/'
checkpoint_dir = './checkpoint/Sony/'
result_dir = './result_Sony/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.nn.depth_to_space(conv10, 2)
    return out


def weights_arr(num_decades):
    #creates an array of 127 values, from 1 - 10^5 (to represent the intensity range)
    weights = np.float32(np.logspace(0, num_decades, 127, endpoint=True, base=10.0))
    #this normalizes the weights
    weights = weights/np.max(weights)
    
    #this prioritizes the lower weight values over the higher weight values
    #coppied from the Restoring Extremely Dark Images in Real Time github, becasue the it wasnt clear I had to flip the values to get it to work
    weights = np.flipud(weights).copy()

    return weights

def amp_val(bins,w_arr,img,m):
    #ensures i dont overwrite my image values, cuz python
    pixel_arr = img.copy()

    #store conditions for each weight range based on the intensity bins.
    pixel_weight_bin_dict = {}
    #goes through every weight bin
    for i in range(len(w_arr)):
        #check the image array nomalize pixel values and sees if thier bk < pixel(x,y) < bk+1
        #choose which bin to put a pixel into
        pixel_weight_bin_dict[w_arr[i]] = (bins[i] <= pixel_arr) & (pixel_arr <bins[i+1])
        # print(pixel_weight_bin_dict)

    #chatgbt line
    """
    pixel_weight_bin_dict = {}
    #goes through every weight bin
    for i in range(len(w_arr)):
        #check the image array nomalize pixel values and sees if thier bk < pixel(x,y) < bk+1
        #choose which bin to put a pixel into
        pixel_weight_bin_dict[w_arr[i]] = (bins[i]<=pixel_arr)&(pixel_arr<bins[i+1])
        # print(pixel_weight_bin_dict)

    find a way to correspond each weight to its pixel value as true or false for each weight bin. like a boolean mask for weight bin that has a array weather a pixel value true or false satisfy the condition
    """
    #creates the pixel weights for the entire image
    image_amp_weights = np.select(pixel_weight_bin_dict.values(), pixel_weight_bin_dict.keys())
    #amplification formula from the paper
    amp = (np.sum(image_amp_weights)*m) / np.sum(img*image_amp_weights)
    
    #since the SID paper only has amplification values from 1-300, these lines are added in to clip the values
    if amp > 300:
        amp = 300.0

    if amp < 1.0:
        amp = 1.0

    return amp


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

#Code was slightly altered to run on Tensorflow 2
sess = tf.compat.v1.Session()
in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

data = {
    'Image_Name': [],
    'Camera_Manufacturer': [],
    'ISO': [],
    'Shutter_Speed': [],
    'GT_Exposure_Ratio': [],
    'Input_PSNR': [],
    'Lamba_PSNR': [],
    'GT_PSNR': [],
    'Input_SSIM': [],
    'Lamba_SSIM': [],
    'GT_SSIM': [],
    'ImageGT_tf': []
}

for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)


    for k in range(len(in_files)):
        
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        data['Image_Name'].append(in_fn)
        data['Camera_Manufacturer'].append("Sony")
        data['ISO'].append(100)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])

        
        #reads inputted raw image
        raw = rawpy.imread(in_path)
        #process the raw image to convert from intensity to a readable image
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        
        
        #Lamba Amplification Value for non-GT Ratio Values
        #creates evenly spaced log bins
        #this allow for more detail and varaitions for better differentiation in the darker regions, but courser variation in the brighter regions where intensity changes are less subtle
        bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
        
        #comptues the array of weights
        #logarimically spaces the weights from 10^0 to 10^5 to cover the intensity values
        weight_arr = weights_arr(5)
        #gets the raw image intensity values
        weight_image = raw.raw_image_visible.astype(np.float32)     
        #normilizes the image inenstiy values
        weight_image = np.maximum(weight_image - 512, 0) / (16383 - 512)
        #computes the Lamba pre-amplification value
        amp_Lamba = amp_val(bins,weight_arr, weight_image, .05)
        #computes the Ground Truth pre-amp value
        GT_ratio = min(gt_exposure / in_exposure, 300)
        
        #applies the pre-amplification values
        #makes sure the amplification values don't go over the normalize values
        input_full_Lamba = np.minimum(np.expand_dims(pack_raw(raw), axis=0) * amp_Lamba, 1.0)
        input_full_GT = np.minimum(np.expand_dims(pack_raw(raw), axis=0) * GT_ratio, 1.0)
        
        
        #reads the test image 
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #increase the dimensions by 1, for input reason
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        #reads the groud truth image
        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        #puts the inputted image in the U-Net pipline
        #output of the Ground Truth Amplification
        output_GT = sess.run(out_image, feed_dict={in_image: input_full_GT})
        #makes sure the image is normalized properly
        output_GT = np.minimum(np.maximum(output_GT, 0), 1)

        #puts the inputted image in the U-Net pipline
        #output of the lamba Amplification
        output_Lamba = sess.run(out_image, feed_dict={in_image: input_full_Lamba})
        #makes sure the image is normalized properly
        output_Lamba = np.minimum(np.maximum(output_Lamba, 0), 1)


        #formats the a image for testing, by removing the dimension -> ie the opposite of the np.expand_dims
        output_GT = output_GT[0, :, :, :]
        output_Lamba = output_Lamba[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)  # scale the low-light image to the same mean of the groundtruth
        
        #inputs the data into the dictionary
        data['Shutter_Speed'].append(in_exposure)
        data['GT_Exposure_Ratio'].append(GT_ratio)
        data['ImageGT_tf'].append(False)


        #compares the outputted images to the ground truth, computes and store the PSNR & SSIM metrics
        #this was completly written by me
        data['GT_PSNR'].append(compare_psnr((gt_full*255).astype('uint8'), (output_GT*255).astype('uint8')))
        data['Input_PSNR'].append(compare_psnr((gt_full*255).astype('uint8'), (scale_full*255).astype('uint8')))
        data['GT_SSIM'].append(compare_ssim((gt_full*255).astype('uint8'), (output_GT*255).astype('uint8'), channel_axis=2))
        data['Input_SSIM'].append(compare_ssim((gt_full*255).astype('uint8'), (scale_full*255).astype('uint8'), channel_axis=2))

        data['Lamba_PSNR'].append(compare_psnr((gt_full*255).astype('uint8'), (output_Lamba*255).astype('uint8')))
        data['Lamba_SSIM'].append(compare_ssim((gt_full*255).astype('uint8'), (output_Lamba*255).astype('uint8'), channel_axis=2))
        
        
        #the old image output used an outdated method and function
        #used the PIL library to create a new method to save the processed images
        Image.fromarray((output_GT * 255).astype('uint8'), mode='RGB').save(result_dir + 'final/%5d_00_%d_out_GT.png' % (test_id, GT_ratio))
        Image.fromarray((output_Lamba * 255).astype('uint8'), mode='RGB').save(result_dir + 'final/%5d_00_%d_out_Lamba.png' % (test_id, GT_ratio))
        Image.fromarray((scale_full * 255).astype('uint8'), mode="RGB").save( result_dir + 'final/%5d_00_%d_scale.png' % (test_id, GT_ratio))
        Image.fromarray((gt_full * 255).astype('uint8'), mode="RGB").save( result_dir + 'final/%5d_00_%d_gt.png' % (test_id, GT_ratio))
        
#saves the testing data to csv    
df = pd.DataFrame(data)
df.to_csv('Sony_data.csv', index=False)

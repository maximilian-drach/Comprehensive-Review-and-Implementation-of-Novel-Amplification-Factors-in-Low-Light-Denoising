from __future__ import division
import rawpy
from exiftool import ExifToolHelper
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


checkpoint_dir = './checkpoint/Sony/'
Canon_dir = './dataset/Canon/'
# Canon_dir = '../Learning-to-See-in-the-Dark/dataset/Canon/'
result_dir = './result_Canon/'


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
    
    #no cliping, becasue we are trying to measure high GT Ratios
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
    'GT_Exposure_Ratio':[],
    'Input_PSNR': [],
    'Lamba_PSNR': [],
    'GT_PSNR': [],
    'Input_SSIM': [],
    'Lamba_SSIM': [],
    'GT_SSIM': [],
    'ImageGT_tf': []
}


#############################################################
#gets all the files in the Canon Dataset
all_files = os.listdir(Canon_dir)
images = []
Cannon_image_files = []
prev_ss = 0
#loops through every file in the canon dataset directory
for file in all_files:
    #this uses the exiftools to parse the metadata in the image
    with ExifToolHelper() as et:
        #gets the photo meta data
        d = et.get_metadata(Canon_dir + file)
        #from the metadata stores the exposure time
        ss = d[0]["EXIF:ExposureTime"]

        #test to see if the previous exposure time was less than the current exopsure time
        #if true, then it mean the current image is a ground truth image and it is the start of a new set
        if(ss > prev_ss):
            # print(images)
            #appends the scene set of images to the big dataset of all the sublist of scenes
            Cannon_image_files.append(images)
            #clears the image dataset for a new image
            images = []

        
        prev_ss = ss
        #appends the image file to the new schene sublist
        images.append(file)
#since the last sublist schene isnt append in the loop this appends the last scene to the datalist
Cannon_image_files.append(images)
#since the first inputted value is a blank list, it takes it out of the dataset
Cannon_image_files = Cannon_image_files[1:]

#scene counter
i = 1
for image_files in Cannon_image_files:
    #since the first image in a scenes list is the GT image, stores it as grond truth
    gt_file_name = image_files[0]
    gt_path = Canon_dir + gt_file_name

    #parse the image metadata
    with ExifToolHelper() as et:
            d = et.get_metadata(gt_path)
            #gets the image shutter speed
            gt_exposure = d[0]["EXIF:ExposureTime"]
            #gets the image ISO value
            gt_ISO = d[0]["EXIF:ISO"]
            #gets the camera manufactuer
            camera_make = d[0]["EXIF:Make"].split()[0]
    #appends the data to the dictionary values
    data['Image_Name'].append(gt_file_name)
    data['Camera_Manufacturer'].append(camera_make)
    data['ISO'].append(gt_ISO)
    data['Shutter_Speed'].append(gt_exposure)
    data['GT_Exposure_Ratio'].append(1.0)   
    data['ImageGT_tf'].append(True)
    data['Lamba_PSNR'].append(0.0)
    data['GT_PSNR'].append(0.0)
    data['Lamba_SSIM'].append(0.0)
    data['GT_SSIM'].append(0.0)
    data['Input_PSNR'].append(0.0)
    data['Input_SSIM'].append(0.0)

    #takes out the first GT image in the scene image list
    for image in image_files[1:]:
        in_exposure = 0
        in_path = Canon_dir + image
        #parse the image metadata
        with ExifToolHelper() as et:
            d = et.get_metadata(in_path)
            #gets the image shutter speed
            in_exposure = d[0]["EXIF:ExposureTime"]
            #gets the image ISO value
            in_ISO = d[0]["EXIF:ISO"]
            #gets the camera manufactuer
            camera_make = d[0]["EXIF:Make"].split()[0]

        
        print(image)
        #reads inputted raw image
        raw = rawpy.imread(in_path)
        #process the raw image to convert from intensity to a readable image
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        

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
        print("GT Exposure Ratio: ", gt_exposure / in_exposure, ", ISO: ", in_ISO)
        #computes the Ground Truth pre-amp value, no max 300, because we are trying to test higher values
        GT_ratio = gt_exposure / in_exposure
        
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
        data['Image_Name'].append(image)
        data['Camera_Manufacturer'].append(camera_make)
        data['ISO'].append(in_ISO)
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
        Image.fromarray((output_GT * 255).astype('uint8'), mode='RGB').save(result_dir + 'final/%d_%d_%d_out_GT.png' % (i, in_ISO, GT_ratio))
        Image.fromarray((output_Lamba * 255).astype('uint8'), mode='RGB').save(result_dir + 'final/%d_%d_%d_out_Lamba.png' % (i, in_ISO, GT_ratio))
        Image.fromarray((scale_full * 255).astype('uint8'), mode="RGB").save( result_dir + 'final/%d_%d_%d_scale.png' % (i, in_ISO, GT_ratio))
        Image.fromarray((gt_full * 255).astype('uint8'), mode="RGB").save( result_dir + 'final/%d_%d_%d_gt.png' % (i, in_ISO, GT_ratio))
    #increments to the next scene value
    i += 1


df = pd.DataFrame(data)
# Save DataFrame to CSV
df.to_csv('Canon_data.csv', index=False)

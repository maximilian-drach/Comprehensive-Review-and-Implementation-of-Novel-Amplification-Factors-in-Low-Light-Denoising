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
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd


checkpoint_dir = './checkpoint/Sony/'
Nikon_dir = './dataset/Nikon/'
# Nikon_dir = '../Learning-to-See-in-the-Dark/dataset/Nikon/'
result_dir = './result_Nikon/'


#gets all the files in the dir
all_files = os.listdir(Nikon_dir)
#this segements the code by shutter speed
images = []
Nikon_image_files = []
prev_ss = 0
#loops through every file in the nikon dataset directory
for file in all_files:
    #this uses the exiftools to parse the metadata in the image
    with ExifToolHelper() as et:
        #gets the photo meta data
        d = et.get_metadata(Nikon_dir + file)
        ss = d[0]["EXIF:ExposureTime"]

        #test to see if the previous exposure time was less than the current exopsure time
        #if true, then it mean the current image is a ground truth image and it is the start of a new set
        if(ss > prev_ss): #if the shutter speed is higher than the previous then its the start of a new image
            #appends the scene set of images to the big dataset of all the sublist of scenes
            Nikon_image_files.append(images)
            images = []

        
        prev_ss = ss
        images.append(file)

#since the last sublist schene isnt append in the loop this appends the last scene to the datalist
Nikon_image_files.append(images)
# print(Nikon_image_files)
Nikon_image_files = Nikon_image_files[1:]
#the first two picture are dark frame calibration picture, because it is more accurate at the channel black level
black_frame1 = rawpy.imread(Nikon_dir + Nikon_image_files[0][0])
black_frame2 = rawpy.imread(Nikon_dir + Nikon_image_files[0][1])
#gets the average pixel values over both images
black_level = np.average(black_frame1.raw_image) + np.average(black_frame2.raw_image) / 2
#takes the black frames out of the image dataset
Nikon_image_files = Nikon_image_files[1:]
# print(Nikon_image_files)

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
    
    if amp < 1.0:
        amp = 1.0

    return amp


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    #change the black level error, because its not 512 and different camera
    #after testing with the metadata black channel level and the emperical black channel level, the emperical black level had higher PSNR & SSIM scores, so we will be using that as my black channel level.
    im = np.maximum(im - int(black_level), 0) / (16383 - int(black_level))  # subtract the black level
    # im = np.maximum(im - 0, 0) / (16383 - 0)  # subtract the black level

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



#this are the absolute ground truth images, with ISO 100
GT_100_files = ['_AJN8217.NEF', '_AJN8263.NEF', '_AJN8351.NEF']

#scene number
j = 0
for image_files in Nikon_image_files:
    # test the first image in each sequence
    for i in range(len(image_files)):
        image = image_files[i]
        #checks if its the first image in a start of a scene, ie gt image
        if (i == 0):
            #local gt image, ie its the ground truth image but it might not be the true GT image where the ISO 100
            gt_file_name = image
            gt_path = Nikon_dir + gt_file_name
            #gets the metadata
            with ExifToolHelper() as et:
                    d = et.get_metadata(gt_path)
                    gt_exposure = d[0]["EXIF:ExposureTime"]
                    gt_ISO = d[0]["EXIF:ISO"]
                    camera_make = d[0]["EXIF:Make"].split()[0]
            #stores the metadata
            data['Image_Name'].append(image)
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

            #checks to see if the inputed GT image is ISO 100, ie a true GT image with the most minimal noise
            if image in GT_100_files:
                #increments the scenario number, ISO 100 GT is the start of a new scenario
                j+=1
                gt100_file_name = image
                gt100_path = Nikon_dir + gt100_file_name
        
        #Not GT images
        else:
            
            in_exposure = 0
            in_path = Nikon_dir + image
            #gets meta data
            with ExifToolHelper() as et:
                d = et.get_metadata(in_path)
                in_exposure = d[0]["EXIF:ExposureTime"]
                in_ISO = d[0]["EXIF:ISO"]
                camera_make = d[0]["EXIF:Make"].split()[0]
            
            print(image)
            raw = rawpy.imread(in_path)
            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
           
            
            #Lamba Amplification Value for non-GT Ratio Values
            #creates evenly spaced log bins
            #this allow for more detail and varaitions for better differentiation in the darker regions, but courser variation in the brighter regions where intensity changes are less subtle
            bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0

            #comptues the array of weights
            #logarimically spaces the weights from 10^0 to 10^5 to cover the intensity values
            weight_arr = weights_arr(5)
            #gets the raw image intensity values
            weighted_image = raw.raw_image_visible.astype(np.float32)

            #need 7 because that is the found black channel level from empirical black frame
            #black level is approx 7.5 rounded to 8
            #also nomalizes the pixel intensity values
            weighted_image = np.maximum(weighted_image - int(black_level), 0) / (16383 - int(black_level))  # subtract the black level
            # weighted_image = np.maximum(weighted_image - 0, 0) / (16383 - 0)
            
            #computes the Lamba pre-amplification value
            amp_Lamba = amp_val(bins,weight_arr, weighted_image,.05)
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

            #reads the ground truth image at ISO 100 to get the real GT image
            gt100_raw = rawpy.imread(gt100_path)
            im = gt100_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            #make the comparison image between the ISO 100 ground truth, not higher ISO because those have more induced noise from the artifical amplification
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
            Image.fromarray((output_GT * 255).astype('uint8'), mode='RGB').save(result_dir + 'final/%d_%d_%d_out_GT.png' % (j, in_ISO, GT_ratio))
            Image.fromarray((output_Lamba * 255).astype('uint8'), mode='RGB').save(result_dir + 'final/%d_%d_%d_out_Lamba.png' % (j, in_ISO, GT_ratio))
            Image.fromarray((scale_full * 255).astype('uint8'), mode="RGB").save( result_dir + 'final/%d_%d_%d_scale.png' % (j, in_ISO, GT_ratio))
            Image.fromarray((gt_full * 255).astype('uint8'), mode="RGB").save( result_dir + 'final/%d_%d_%d_gt.png' % (j, in_ISO, GT_ratio))
        # df = pd.DataFrame(data)
        # print(data)
        # print(df)


df = pd.DataFrame(data)
# Save DataFrame to CSV
df.to_csv('Nikon_data.csv', index=False)


    

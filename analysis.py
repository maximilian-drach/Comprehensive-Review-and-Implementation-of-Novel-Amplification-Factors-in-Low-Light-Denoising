# from __future__ import division
import rawpy
import exiftool
from exiftool import ExifToolHelper
import os, scipy.io
# import tensorflow as tf
import numpy as np
import rawpy
import glob
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import sys
# Display all columns
pd.set_option('display.max_columns', None)



canon_df = pd.read_csv("Canon_data.csv")
#reads only the information from the tested images, not the GT images
canon_test_df = canon_df[canon_df['ImageGT_tf'] == False]
print('Canon Total Average Metrics')
print(canon_test_df[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))

print("GT-Ratio >= 100 (SID Control) Metrics")
#this controls for GT Ratios less than 100 to have a more comparison
canon_test_df = canon_df[canon_df['GT_Exposure_Ratio'] >= 100]
print('Canon GT-Ratio Adjusted Average Metrics GT-Ratio >= 100 (SID Control)')
print(canon_test_df[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))


canon_test_df_gt300 = canon_test_df[canon_test_df['GT_Exposure_Ratio'] > 300]
canon_test_df_lt300 = canon_test_df[canon_test_df['GT_Exposure_Ratio'] <= 300]
print('Canon GT Ratio > 300 Average Metrics')
print(canon_test_df_gt300[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))
print('Canon GT Ratio <= 300 Average Metrics')
print(canon_test_df_lt300[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))


print("\n")
nikon_df = pd.read_csv("Nikon_data.csv")
#reads only the information from the tested images, not the GT images
nikon_test_df = nikon_df[nikon_df['ImageGT_tf'] == False]
print('Nikon Total Average Metrics')
print(nikon_test_df[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))

print("GT-Ratio >= 100 (SID Control) Metrics")
#this controls for GT Ratios less than 100 to have a more comparison
nikon_test_df = nikon_df[nikon_df['GT_Exposure_Ratio'] >= 100]
print('Nikon GT-Ratio Adjusted Average Metrics GT-Ratio >= 100 (SID Control)')
print(nikon_test_df[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))


nikon_test_df_gt300 = nikon_test_df[nikon_test_df['GT_Exposure_Ratio'] > 300]
nikon_test_df_ltq300 = nikon_test_df[nikon_test_df['GT_Exposure_Ratio'] <= 300]
print('Nikon GT Ratio > 300 Average Metrics')
print(nikon_test_df_gt300[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))
print('Nikon GT Ratio <= 300 Average Metrics')
print(nikon_test_df_ltq300[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))



nikon_test_df_ISO500 = nikon_test_df[nikon_test_df['ISO'] >= 500]
print('Nikon Average Metrics ISO >= 500')
print(nikon_test_df_ISO500[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))

nikon_test_df_ISO1000 = nikon_test_df[nikon_test_df['ISO'] >= 1000]
print('Nikon Average Metrics ISO >= 1000')
print(nikon_test_df_ISO1000[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))

nikon_test_df_ISO200 = nikon_test_df[nikon_test_df['ISO'] <= 200]
print('Nikon Average Metrics ISO <= 200')
print(nikon_test_df_ISO200[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))
print("\n")


sony_df = pd.read_csv("Sony_data.csv")
sony_test_df = sony_df[sony_df['ImageGT_tf'] == False]
print('Sony Total Average Metrics')
print(sony_test_df[['Lamba_PSNR', 'GT_PSNR', 'Input_PSNR', 'Lamba_SSIM', 'GT_SSIM', 'Input_SSIM']].mean(axis=0))
print("\n*******************************************************************\n")
end_dict = {"cr3":"Canon","nef":"Nikon", "arw":"Sony"}

def check_path(file_path):
    if os.path.exists(file_path) == False:
        print(f"Your path file {file_path} does not exisit for the pixel array statistics")
        print("No Pixel Array Statistic will be outputted")
        sys.exit(1)
    else:
        if file_path[-3:] not in end_dict :
            print(f"image file not supported!")
            print("No Pixel Array Statistic will be outputted")
            sys.exit(1)


if len(sys.argv) == 4:
    check_path(sys.argv[1])
    check_path(sys.argv[2])
    check_path(sys.argv[3])
    print("Pixel Array Statstics Comparison")
    img = rawpy.imread(sys.argv[1])
    
    print(f"{end_dict[sys.argv[1][-3:].lower()]} Image Sensor Array Info")
    print("Number of Colors: ", img.num_colors)
    print("Black Level per Channel: ", img.black_level_per_channel)
    print("Bayer Pattern: \n", img.raw_pattern)


    img = rawpy.imread(sys.argv[2])
    print(f"{end_dict[sys.argv[2][-3:].lower()]} Image Sensor Array Info")
    print("Number of Colors: ", img.num_colors)
    print("Black Level per Channel: ", img.black_level_per_channel)
    print("Bayer Pattern: \n", img.raw_pattern)



    img = rawpy.imread(sys.argv[3])
    print(f"{end_dict[sys.argv[3][-3:].lower()]} Image Sensor Array Info")
    print("Number of Colors: ", img.num_colors)
    print("Black Level per Channel: ", img.black_level_per_channel)
    print("Bayer Pattern: \n", img.raw_pattern)

elif len(sys.argv) >= 2:
    if sys.argv[1].lower() == "nope":
        print("NO Pixel Array Statstics Comparison")
    else:
        print("No Pixel Array Statistic will be outputted")
        sys.exit(1)

else:
    
    check_path("./dataset/Nikon/_AJN8217.NEF")
    check_path("./dataset/Canon/IMG_0048.CR3")
    check_path("./dataset/Sony/long/00001_00_10s.ARW")

    print("Pixel Array Statstics Comparison")
    img = rawpy.imread("./dataset/Canon/IMG_0048.CR3")
    print("Canon Image Sensor Array Info")
    print("Number of Colors: ", img.num_colors)
    print("Black Level per Channel: ", img.black_level_per_channel)
    print("Bayer Pattern: \n", img.raw_pattern)


    img = rawpy.imread("./dataset/Nikon/_AJN8217.NEF")
    print("Nikon Image Sensor Array Info")
    print("Number of Colors: ", img.num_colors)
    print("Black Level per Channel: ", img.black_level_per_channel)
    print("Bayer Pattern: \n", img.raw_pattern)



    img = rawpy.imread("./dataset/Sony/long/00001_00_10s.ARW")
    print("Sony Image Sensor Array Info")
    print("Number of Colors: ", img.num_colors)
    print("Black Level per Channel: ", img.black_level_per_channel)
    print("Bayer Pattern: \n", img.raw_pattern)

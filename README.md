# Comprehensive Review and Implementation of Novel Amplification Factors in Low-Light Denoising

## How to Run the Experiment

1. **Download Pretrained Model Weights:**
   Run the download_models.py script. This will download the pretrained model weights into the checkpoint folder.

2. **Download Datasets:**
   - The SID dataset comes from the "Learning to See in the Dark" paper.
     - [SID Sony Dataset](https://storage.googleapis.com/isl-datasets/SID/Sony.zip) (25 GB)
   - Datasets collected by the author for testing:
     - [Nikon Dataset](https://purdue0-my.sharepoint.com/:f:/g/personal/mdrach_purdue_edu/EoVPnKf73EdEkOeCDegLhpwBoHOtytsfrZqkXRHtFh-w_A?e=W5YRbK)
     - [Canon Dataset](https://purdue0-my.sharepoint.com/:f:/g/personal/mdrach_purdue_edu/EnFeIP1vbORNq-lEmeeP71sBJiFlTHOja0vSb3zkoLAV-A?e=SEfwGZ)

3. **Dataset Preparation:**
   - Unzip the datasets and place them in the proper folder structure:
     - Sony -> ./dataset/Sony (two folders: long and short)
     - Canon -> ./dataset/Canon
     - Nikon -> ./dataset/Nikon

4. **Install Required Libraries:**
   Make sure all the necessary libraries are installed. Use the following commands for installation:

   pip install rawpy
   pip install exiftool
   pip install scipy
   pip install tensorflow==2.9.0
   pip install tf_slim
   pip install numpy
   pip install Pillow
   pip install scikit-image
   pip install pandas

5. **Run Test Scripts**
   - run test_Sony.py, test_Nikon.py, test_Canon.py to test the amplification factors and the pipeline
   - outputs the enhanced image results in result_Canon/final, result_Nikon/final, result_Nikon/final
   - returns a .csv file as the output of the testing rusults

6. **Run the Analysis Script**
   - run analysis.py file_path1 file_path2 file_path3
   - input the paths for the raw photos you want to see the pixel array statistics for
   OR
   - run analysis.py nope
   - put Nope after the file and it will not do any pixel array comparison
   OR
   - run analysis.py 
   - if no paths it will make a default decision

   - outputs to the commandline an analysis of the resulting testing data

## Code File Description

1. download_models.py, test_Sony.py, checkpoint/Sony/checkpoint, checkpoint/Sony/model.chkpt.index, dataset/Sony/Sony_test_list.txt, dataset/Sony/Sony_train_list.txt, dataset/Sony/Sony_val_list.txt all orginally came from the Learn to See in the Dark Github (https://github.com/cchen156/Learning-to-See-in-the-Dark)

2. 
Test_Sony.py:
I modified it to run on the presently available Tensorflow 2.9.0, to test the Lamba Amplification factor, and store the values of the test.
test_Sony.py (lines 157-168) -> I modtified the code slightly to use a wrapper function built by Tensorflow that enables older Tensorflow 1.X.X to run on Tensorflow 2.X.X

test_Sony.py (lines 173-186) -> create a dictionary to store the PSNR & SSIM testing metrics

test_Sony.py (lines 223-229) -> These lines of code create the Lamba Amplification Factor as outline in research paper "Restoring Extremely Dark Images in Real Time"

test_Sony.py (lines 88-137) -> These are the functions written by me to create the Lamba Amplification Factor from his reserach paper
* lines 96 was added in from (https://github.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time) because my Lamba Amplification was not working and I need to double check what they had done differently than my own implimentation 
* look at the code to see the comments about how I implimented the paper (lines 88-137)

test_Sony.py (lines 270-291) -> alter the code to compute the SSIM & PSNR values for each inputted image, append the metrics values to the data dictionary, and showcase the end resulting image after running the image through the pipline. 
test_Sony.py (lines 294-295) ->Finally I converted the dictionary to a pandas dataframe and stored the data as a csv file.

test_Canon.py: 
Line are 1 - 120 are the same from the test_Sony.py and only slightly altered to test the Canon Dataset

test_Canon.py (lines 122-123) -> I've taken out the limiting of the amplification, becuase I want to test GT amplification factors > 300

test_Canon.py (lines 179-208) -> these line parse the Canon dataset images and segment them into the different schenes. It does this by checking if the previoius files exposure time was less then the current exposure time. If its true then it means a new ground truth image was caputured and it becomes the start of a new schene. LOOK at the comments for more detail about what the code is doing.

test_Canon.py (lines 212-238) -> stores the ground truth image metadata in the dictionary

test_Canon.py (line 282) -> GT_ratio is not capped at 300, but instead can have a higher amplification vlaue
* the rest of the code is the same as test_Sony.py, formatted slightly differnt to store the data for the Canon dataset test metrics


test_Nikon.py: 
Same general code from the test_Canon.py, but with some extra changes to account for ISO difference between scenes and other small changes

test_Nikon.py (lines 53-56) -> In my testing the Nikon Black Channel Level was 0, but my empirically measured black channel level from 2 dark frames was 7. After testing with the metadata black channel level and the empirical level the emperical black level had higher PSNR & SSIM scores, so we will be using that as my black channel level. I empirically found the black channel level by getting the average pixel value of two black frames, ie lens cap on the camera, fastest shutter speed possible, lowest ISO setting. 

test_Nikon.py (lines 170) -> had to change the black level error, because the Nikon had a different level than both the Canon and Sony

test_Nikon.py (lines 218) -> this stores the absolute GT image file names, which are the images with ISO 100 and hence the lowest amount of noise 

test_Nikon.py (lines 227 - 256) -> Slight change from the test_Canon.py to account for two different ground truth images. gt100 is the low noise ground truth taken at ISO100, while the other ground truths are taken at higher ISO levels


test_Nikon.py (lines 298) -> I use the local GT exposure (not GT ISO 100) to calcuate the amplification, because the inputted photo is taken relative with a higher amplification value


test_Nikon.py (lines 311) -> Since I want to compare the quality of the image to the best quality ground truth, i use the ground truth image at ISO 100 to compare the results of the pipline output


analysis.py:
!!!Make sure to have all 3 metric csv's  AND the photo datasets!!!!

This is simple script that you can run, after your csv files have been created to preform an analysis on the csv metric data
It also shows how I test the Black Channel Level & Saw that the Bayer Arrays were the same


## Datasets
I used the SID dataset from Learning to See in the Dark github and built the other two datasets I used.
- The SID dataset comes from the "Learning to See in the Dark" paper.
     - [SID Sony Dataset](https://storage.googleapis.com/isl-datasets/SID/Sony.zip) (25 GB)
   - Datasets collected by the author for testing:
     - [Nikon Dataset](https://purdue0-my.sharepoint.com/:f:/g/personal/mdrach_purdue_edu/EoVPnKf73EdEkOeCDegLhpwBoHOtytsfrZqkXRHtFh-w_A?e=W5YRbK)
     - [Canon Dataset](https://purdue0-my.sharepoint.com/:f:/g/personal/mdrach_purdue_edu/EnFeIP1vbORNq-lEmeeP71sBJiFlTHOja0vSb3zkoLAV-A?e=SEfwGZ)











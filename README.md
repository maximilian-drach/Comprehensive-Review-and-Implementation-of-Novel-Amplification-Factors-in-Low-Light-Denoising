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

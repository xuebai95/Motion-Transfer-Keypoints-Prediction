# Motion-Transfer-Keypoints-Prediction

## Description of the repository:
This is a repo containing files related to keypoint prediction and video generation using RNN/VAE/VRNN in the First Order Motion Model (FOMM) pipeline. 

## Directory Contents:
**The Generated_Videos folder** contains sample videos for the VoxCeleb dataset that have been generated in reconstruction and transfer modes using a prediction horizon of 12 video frames.
In each video, there are 4 boxes as follows: 

1.  The source image 

2. Driving video frames with keypoints

3. Generated video from FOMM pipeline

4. Generated video from FOMM with keypoints prediction using VRNN (the red box indicates the predicted frames)

**The Keypoints_Prediction folder** contains Jupyter notebooks that can be used to generate predictions using RNN/VAE/VRNN in the FOMM pipeline in either reconstruction or transfer mode for the VoxCeleb dataset.
The naming convention followed for the notebooks is:
"Full_Pipeline_{Deep Learning Prediction Network}\_VoxCeleb\_{Mode}_mode" where the Deep Learning Prediction Network can be RNN/VAE/VRNN and Mode can be reconstruction or transfer.

The Training_Prediction subfolder contains the following:

1. The subfolder FOMM contains Voxceleb data files and various functions related to FOMM inference.
The files are partially sourced from the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model.

2. The subfolder PREDICTOR contains prediction functions using RNN, VAE and VRNN.
The files are partially sourced from:
https://github.com/google-research/google-research/tree/master/video_structure, https://github.com/msmbuilder/vde.

The config subfolder contains the yaml file for the VoxCeleb dataset.

The checkpoints subfolder contains the trained RNN/VAE/VRNN keypoints prediction models using prediction horizons of 6 or 12 for RNN/VAE/VRNN.
Checkpoints for RNN/VAE/VRNN are named as "{Deep Learning Network}\_3883videos_vox_{# input frames}_{# output frames}" where {# input frames} and {# output frames} can be 6/12 indicates types of prediction.

The log subfolder is the directory for saving generated videos.

The two pickle files are the keypoints corresponding to 44 VoxCeleb videos during inference for source image and driving video frames.
## Checkpoints for the FOMM model and keypoints 
Checkpoints for the FOMM model trained on the VoxCeleb dataset can be found under this google drive link. 
https://drive.google.com/drive/folders/1pachVtWHibzDi3E61jUmqFfz2hVxA1GX?usp=drive_link.

This file has been sourced using the link in the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model.

To run this file in the attached Jupyter notebooks, please copy the checkpoint file to the following path "Training_Prediction/FOMM/Trained_Models/".

The keypoints corresponding to 3883 VoxCeleb videos which can be used to train the RNN/VAE/VRNN can be found with the same google drive link.

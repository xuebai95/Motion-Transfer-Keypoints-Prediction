# Motion-Transfer-Keypoints-Prediction

# Description of the repository:
This is a repo containing files for keypoint prediction and video generation using RNN/VAE/VRNN in the First Order Motion Model (FOMM) pipeline. 

# Directory Contents:
The Generated Videos folder contains sample videos for the VoxCeleb dataset that have been generated in reconstruction and transfer mode using a prediction horizon of 12.
In each video, there are 4 boxes in a row, 1st box represents the soure image, 2nd box represents driving video frames with keypoint, 3rd box represents generated video from FOMM pipeline and 4th box represents generated video from FOMM with keypoints prediction using VRNN.

The keypoints_Prediction folder contains Jupyter notebooks that can be used to run prediction using the RNN/VAE/VRNN in the FOMM pipeline in either reconstruction or transfer mode for the VoxCeleb dataset. This involves training the predictor and then performing inference. 
The naming convention followed for the notebooks is as below:
"Full_Pipeline_{Deep Learning Prediction Network}\_VoxCeleb\_{Mode}_mode" where the Deep Learning Network 

The Training_Prediction folder under keypoints_Prediction contains the Voxceleb data file and functions related to FOMM inference in the subfolder FOMM and prediction using RNN, VAE and VRNN in the subfolder PREDICTOR.

The FOMM subfolder files are partially sourced from the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model

The PREDICTOR subfolder contains files for prediction using RNN, VAE and VRNN (partially sourced from https://github.com/google-research/google-research/tree/master/video_structure).

The config folder under keypoints_Prediction contains the yaml file of VoxCeleb dataset.

The checkpoints folder under keypoints_Prediction contains the trained RNN/VAE/VRNN keypoints prediction models using prediction horizon of 6 or 12 for RNN/VRNN and 5 or 15 for VAE.
Checkpoints for RNN/VAE/VRNN are named as "{Deep Learning Network}\_3883videos_vox_{# input frames}_{# output frames}" where {# input frames} and {# output frames} can be 6/12 indicates types of prediction.

The log folder under keypoints_Prediction is the directory for saving generated videos.

The two pickle files under keypoints_Prediction are the keypoints corresponding to 44 VoxCeleb videos during inference for source image and driving video frames.
# Checkpoints for FOMM model and keypoints 
Checkpoints for the FOMM model trained on the VoxCeleb dataset can be found under this google drive link. 
https://drive.google.com/drive/folders/1pachVtWHibzDi3E61jUmqFfz2hVxA1GX?usp=drive_link

This file has been sourced using the link in the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model

To run this file in the attached Jupyter notebooks, please copy the checkpoint file to the following path "Training_Prediction/FOMM/Trained_Models/" 

The keypoints corresponding to 3883 VoxCeleb videos which can be used to train the RNN/VAE/VRNN can be found with the same google drive link.

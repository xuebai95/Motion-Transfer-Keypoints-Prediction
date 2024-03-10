# Motion-Transfer-Keypoints-Prediction

# Description of the repository:
This is a repo containing files for keypoint prediction using RNN/VAE/VRNN in the First Order Motion Model (FOMM) pipeline. 

# Directory Contents:
The Generated Videos folder contains sample videos for the VoxCeleb dataset that have been generated in reconstruction and transfer mode using a prediction horizon of 12.

The Training_Prediction folder contains the Voxceleb data file and functions related to FOMM in the subfolder FOMM and prediction using RNN, VAE and VRNN in the subfolder PREDICTOR.
The FOMM subfolder files are sourced from the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model

The PREDICTOR subfolder contains files for prediction using RNN, VAE and VRNN (sourced from https://github.com/google-research/google-research/tree/master/video_structure).

The config folder contains the yaml file of VoxCeleb dataset.

The log folder is the directory for saving generated videos.

The jupyter files are named as "Full_Pipeline_{Deep Learning Network}\_VoxCeleb\_{Mode}_mode" where the Deep Learning Network used for keypoint prediction can be RNN/VAE/VRNN, and the mode can be set to reconstruction or transfer.

# Checkpoints for FOMM model and keypoints 
Checkpoints for RNN/VAE/VRNN are named as "{Deep Learning Network}_3883videos_vox_{# input frames}_{# output frames}" where {# input frames} and {# output frames} can be 6/12 indicates types of prediction.

Checkpoints for the FOMM model trained on the VoxCeleb dataset can be found under this google drive link. 
https://drive.google.com/drive/folders/1pachVtWHibzDi3E61jUmqFfz2hVxA1GX?usp=drive_link

This file has been sources from the link in the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model

To run this file in the attached Jupyter notebooks, please copy the checkpoint file to the following path "Training_Prediction/FOMM/Trained_Models/" 
The keypoints corresponding to 3883 VoxCeleb videos which can be used to train the RNN/VAE/VRNN can be found with the same google drive link.

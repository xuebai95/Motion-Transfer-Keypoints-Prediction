# Motion-Transfer-Keypoints-Prediction
This is a repo for full pipeline with keypoints prediction using RNN/VAE/VRNN in reconstruction/transfer mode for VoxCeleb dataset

The jupyter files are named as "Full_Pipeline_{Deep Learning Network}_VoxCeleb_{Mode}" where Deep Learning Network can be RNN/VAE/VRNN, and mode can be reconstruction mode/ transfer mode.

Checkpoints for RNN/VAE/VRNN are named as "{Deep Learning Network}_3883videos_vox_{# input frames}_{# output frames}" where {# input frames} and {# output frames} can be 6/12 indicates types of prediction.

Checkpoints of VoxCeleb dataset in FOMM pipeline can be found with google drive link:
https://drive.google.com/drive/folders/1pachVtWHibzDi3E61jUmqFfz2hVxA1GX?usp=drive_link

To run this file in the given jupyter, please add this checkpoint file under path "Training_Prediction/FOMM/Trained_Models/"

The keypoints for 3883 VoxCeleb train videos can be found with the same google drive link.

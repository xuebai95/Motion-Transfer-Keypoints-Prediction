import os   
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from Training_Prediction.FOMM.Source_Model.logger import Logger, Visualizer
import numpy as np
import imageio
from Training_Prediction.FOMM.Source_Model.sync_batchnorm import DataParallelWithCallback
from Training_Prediction.FOMM.Source_Model.modules.RNN_prediction_module import PredictionModule
from Training_Prediction.FOMM.Source_Model.augmentation import SelectRandomFrames, SelectFirstFrames_two, VideoToTensor
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from Training_Prediction.FOMM.Source_Model.frames_dataset import FramesDataset
import tensorflow.compat.v1 as tf
from Training_Prediction.PREDICTOR.Source_Model.VRNN import build_vrnn, get_config
import pickle
from Training_Prediction.PREDICTOR.Source_Model.VRNN_prediction import VRNN_predict
import gc
import pickle

class KPDataset(Dataset):
    """Dataset of detected keypoints"""

    def __init__(self, keypoints_array, num_frames):
        self.keypoints_array = keypoints_array
        self.transform = SelectRandomFrames(consequent=True, number_of_frames=num_frames)

    def __len__(self):
        return len(self.keypoints_array)

    def __getitem__(self, idx):
        keypoints = self.keypoints_array[idx]
        selected = self.transform(keypoints)
        selected = {k: np.concatenate([v[k][0] for v in selected], axis=0) for k in selected[0].keys()}
        return selected

class KPDataset_two(Dataset):
    """Dataset of detected keypoints"""

    def __init__(self, keypoints_array):
        self.keypoints_array = keypoints_array
        self.transform = SelectFirstFrames_two(number_of_frames=24)

    def __len__(self):
        return len(self.keypoints_array)

    def __getitem__(self, idx):
        keypoints = self.keypoints_array[idx]
        selected = self.transform(keypoints)
        return selected

def get_data_from_dataloader_60(loader,config):
    result = []
    mean_pkl = []
    var_pkl = []
    for x in loader:
        #print(x['jacobian'].shape)
        mean_pkl.append(x['value'])
        var_pkl.append(x['jacobian'])
        for k in range(x['value'].shape[0]):
            for i in range(config['prediction_params']['num_frames']):
                curr_data = []
                for j in range(10):
                    curr_data.append(x['value'][k][i][j][0])
                    curr_data.append(x['value'][k][i][j][1])
                    curr_data.append(x['jacobian'][k][i][j][0][0])
                    curr_data.append(x['jacobian'][k][i][j][1][1])
                    curr_data.append(x['jacobian'][k][i][j][0][1])
                    curr_data.append(x['jacobian'][k][i][j][1][0])
                curr_data = np.asarray(curr_data)
                result.append(curr_data)
        #result.append(data)
    data_pkl = {}
    data_pkl['value'] = np.vstack(mean_pkl)
    data_pkl['jacobian'] = np.vstack(var_pkl)
    pickle.dump(data_pkl,open("my_Bair_train.pkl", "wb"))
    result = np.vstack(result)
    print("helper function data shape is " + str(result.shape))
    return result

def get_data_from_dataloader_60_two(loader,num_frames):
    result = []
    mean_pkl = []
    var_pkl = []
    for x in loader:
        #print(x['jacobian'].shape)
        mean_pkl.append(x['value'])
        for k in range(x['value'].shape[0]):
            for i in range(num_frames):
                curr_data = []
                for j in range(10):
                    curr_data.append(x['value'][k][i][j][0])
                    curr_data.append(x['value'][k][i][j][1])
                curr_data = np.asarray(curr_data)
                result.append(curr_data)
        #result.append(data)
    data_pkl = {}
    data_pkl['value'] = np.vstack(mean_pkl)
    pickle.dump(data_pkl,open("my_Bair_train.pkl", "wb"))
    result = np.vstack(result)
    print("helper function data shape is " + str(result.shape))
    return result

def save_obj(obj, name ):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def prediction(config, generator, kp_detector, checkpoint, log_dir, dataset, mode="VRNN"):
    png_dir = os.path.join(log_dir, 'prediction/png')
    log_dir = os.path.join(log_dir, 'prediction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    print("Extracting keypoints...")

    generator.eval()
    kp_detector.eval()

    keypoints_array = []
    
    prediction_params = config['prediction_params']

    for idx in enumerate(dataloader):
        print("Debugging contents of dataloader - before keypoint extraction")
        # print(idx)

    for it, x in tqdm(enumerate(dataloader)):
        if prediction_params['train_size'] is not None:
            if it > prediction_params['train_size']:
                break
        # 1,3,256,256 for training dtaset driving and souce
        with torch.no_grad():
            keypoints = []
            for i in range(x['video'].shape[2]):
                kp = kp_detector(x['video'][:, :, i])
                kp['value'] = torch.reshape(kp['value'],(kp['value'].shape[0],kp['value'].shape[1]//10,10,kp['value'].shape[2]))
                kp['jacobian'] = torch.reshape(kp['jacobian'],(kp['jacobian'].shape[0],kp['jacobian'].shape[1]//10,10,kp['jacobian'].shape[2],kp['jacobian'].shape[3]))
                kp = {k: v.data.cpu().numpy() for k, v in kp.items()}
                keypoints.append(kp)
            keypoints_array.append(keypoints)
    
    print(keypoints_array[0])


    #save_obj(keypoints_array,"keypoints_array")
    #keypoints_array = load_obj("keypoints_array")
    #print("keypoints saved")
    predictor = PredictionModule(num_kp=config['model_params']['common_params']['num_kp'],
                                 kp_variance='matrix',
                                 **prediction_params['rnn_params']).cuda()

    num_epochs = prediction_params['num_epochs']
    lr = prediction_params['lr']
    bs = prediction_params['batch_size']
    num_frames = prediction_params['num_frames']

    init_frames = 12#prediction_params['init_frames']

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=50)

    kp_dataset = KPDataset(keypoints_array, num_frames=num_frames)

    kp_dataloader = DataLoader(kp_dataset, batch_size=bs)

    if mode == "VRNN":
        print("using VRNN")
        train_data = get_data_from_dataloader_60(kp_dataloader,config)
        data_all = {}
        data_all['value'] = []
        data_all['jacobian'] = []
        data_lee = []
        
        #data_lee = get_data_50(test_dataset)
        data_lee = train_data
        data_lee = np.reshape(data_lee,(-1,config['prediction_params']['num_frames'],10,6))
        data_lee = tf.convert_to_tensor(data_lee)
        cfg = get_config()
        input_keypoint = tf.keras.Input(shape=[config['prediction_params']['num_frames'],10,6],name='keypoint')
        observed_keypoints_stop = tf.keras.layers.Lambda(tf.stop_gradient)(
        input_keypoint)
        vrnn_model = build_vrnn(cfg)
        predicted_keypoints, kl_divergence = vrnn_model(observed_keypoints_stop)
        train_model = tf.keras.Model(inputs=[input_keypoint],outputs=[predicted_keypoints])
        vrnn_coord_pred_loss = tf.nn.l2_loss(
        observed_keypoints_stop - predicted_keypoints)
        # Normalize by batch size and sequence length:
        vrnn_coord_pred_loss /= tf.to_float(
          tf.shape(input_keypoint)[0] * tf.shape(input_keypoint)[1])
        train_model.add_loss(vrnn_coord_pred_loss)
        kl_loss = tf.reduce_mean(kl_divergence)  # Mean over batch and timesteps.
        train_model.add_loss(cfg.kl_loss_scale * kl_loss)
        vrnn_optimizer = tf.keras.optimizers.Adam(lr=cfg.learning_rate, clipnorm = cfg.clipnorm)
        train_model.compile(vrnn_optimizer)
        print("Training VRNN_LEE prediction...")
        train_model.fit(x=data_lee, steps_per_epoch = cfg.steps_per_epoch, epochs=cfg.num_epochs)

    if mode == "RNN":
        
        print("Training RNN prediction...")
        for _ in trange(num_epochs):
            loss_list = []
            for x in kp_dataloader:
                #print(x['value'].shape) // x.value is 128*30*10*2 x.jacobian is 128*30*10*2*2
                #print(x['mean'][0])
                x = {k: v.cuda() for k, v in x.items()}
                gt = {k: v.clone() for k, v in x.items()}
                 # new x with only init_frames 
                #x_value_init = x['value'][:,:init_frames]
                #x_jacobian_init = x['jacobian'][:,:init_frames]
                #x['value'] = x_value_init
                #x['jacobian'] = x_jacobian_init

                #prediction = predictor(x)
                #prediction['jacobian'] = gt['jacobian']
                #loss = sum([torch.abs(gt[k][:, init_frames:] - prediction[k]).mean() for k in x if k=='value'])
                for k in x:
                    x[k][:, init_frames:] = 0

                #without constant
                #x['var'][:,:,:,0,1] = 0
                #x['var'][:,:,:,1,0] = 0
                #x['var'][:,:,:] = 0  
                prediction = predictor(x)
                loss = sum([torch.abs(gt[k][:, init_frames:] - prediction[k][:, init_frames:]).mean() for k in x])


                print(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_list.append(loss.detach().data.cpu().numpy())

            loss = np.mean(loss_list)
            scheduler.step(loss)

    # dataset = FramesDataset(is_train=False, **config['dataset_params'],mode="RNN")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    loss_list = []

    print("make prediciton")
    
    for idx in enumerate(dataloader):
        print("Debugging contents of dataloader - after keypoint extraction")
        #print(idx)
  
    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        frames = init_frames
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            kp_source = kp_detector(x['video'][:, :, 0])
            kp_init_mean = []
            kp_init_jacobian = []
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                kp_init_mean.append(kp_driving['value'])
                kp_init_jacobian.append(kp_driving['jacobian'])
            kp_init_mean = torch.cat(kp_init_mean)
            kp_init_mean = torch.reshape(kp_init_mean,(1,kp_init_mean.shape[0],kp_init_mean.shape[1],kp_init_mean.shape[2]))
            kp_init_jacobian = torch.cat(kp_init_jacobian)
            kp_init_jacobian = torch.reshape(kp_init_jacobian,(1,kp_init_jacobian.shape[0],10,2,2))

            if mode == "RNN":
                kp_init_jacobian[frames:,:,:,:] = 0
                kp_init_mean[frames:,:,:] = 0
            
            kp_init = {"value":kp_init_mean,"jacobian":kp_init_jacobian}
            kp_driving_total = None
            if mode == "RNN":          
                kp_driving_total = predictor(kp_init)
                
            if mode == "VRNN":
                dim, kp_driving_total = VRNN_predict(kp_init,train_model)
            print(kp_driving_total['value'].shape)
            for i in range(kp_driving_total['value'].shape[1]): # 1,30,10,2
                kp_driving = {"value":kp_driving_total['value'][0][i],"jacobian":kp_driving_total['jacobian'][0][i]}
                #print(kp_driving['value'].shape) # 30,10,2
                #print(kp_driving['jacobian'].shape)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                        driving=driving, out=out)
                visualizations.append(visualization)
                if torch.abs(out['prediction'] - driving).mean().cpu().numpy() != 0 and i > frames:
                    loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())
            print("Reconstruction loss: %s" % np.mean(loss_list))
            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))
#0.081514

#15/15
# RNN 99 samples 0.0773
# VRNN 99 samples 0.0771 (learning rate 2e-4)
#10/20
# RNN 97 samples 0.0803
# VRNN 97 samples 0.0793

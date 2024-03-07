import math
import numpy as np
import torch
import tensorflow as tf
import numpy as np

# Generate synthetic sine wave data
'''def generate_sine_wave_data(batch_size, time_steps, num_sine_waves, input_dim, frequencies):
    data = []
    for _ in range(num_sine_waves):
        t = np.linspace(0, 2 * np.pi, time_steps)
        frequency = np.random.choice(frequencies)
        sine_wave = np.sin(frequency * t).reshape((1, time_steps, 1))
        
        # Extend the sine wave across the input dimensions
        sine_wave_data = np.tile(sine_wave, (batch_size, 1, input_dim))
        data.append(sine_wave_data)
    
    return np.concatenate(data, axis=-1)'''


def generate_multiple_sine_wave_sets(batch_size, time_steps, num_sets, num_sine_waves_per_set, frequencies, amplitude_factors):
    data = np.empty((batch_size, num_sets, time_steps, num_sine_waves_per_set))
    
    for i in range(batch_size):
        for j in range(num_sets):
            sine_wave_set = []
            
            for _ in range(num_sine_waves_per_set):
                t = np.linspace(0.5, 2 * np.pi, time_steps)
                frequency = np.random.choice(frequencies)
                
                sine_wave = np.sin(frequency * t).reshape((1, time_steps, 1))
                
                # Adjust the amplitude of the sine wave for all dimensions
                sine_wave *= amplitude_factors[_]
                
                sine_wave_set.append(sine_wave)
            
            data[i, j] = np.concatenate(sine_wave_set, axis=-1)
    
    return np.transpose(data, (0, 2, 1, 3))# Define a function to compute NMSE loss

def compute_nmse_loss(y_true, y_pred):
    numerator = tf.reduce_mean(tf.square(y_true - y_pred))
    denominator = tf.reduce_mean(tf.square(y_true))
    nmse_loss = numerator / denominator
    return nmse_loss

def VRNN_predict(kp_data, VRNN):
    frame_num = 24
    #frame_num = 12 # last line was initial code

    #frame_num = 6
    # initialize VAE
    #print(kp_data)
    original_dim = 50
    order = 2
    forward_step = 5
    changed = False
    iterate = 0
    data_all = kp_data
    first_Lee = True
    
    data = []
    for i in range(frame_num):
        curr_data = []
        for j in range(10):
            curr_data.append(kp_data['value'][0][i][j][0])
            curr_data.append(kp_data['value'][0][i][j][1])
            curr_data.append(kp_data['jacobian'][0][i][j][0][0])
            curr_data.append(kp_data['jacobian'][0][i][j][1][1])
            curr_data.append(kp_data['jacobian'][0][i][j][0][1])
            curr_data.append(kp_data['jacobian'][0][i][j][1][0])
        data.append(curr_data)
    data = np.asarray(data)

    data_lee = np.reshape(data, (-1,int(frame_num),10,6))
    data_np = np.zeros((data_lee.shape[0],data_lee.shape[1],data_lee.shape[2],data_lee.shape[3]))
    for i in range(data_lee.shape[0]):
        for j in range(data_lee.shape[1]):
            for k in range(data_lee.shape[2]):
                for l in range(data_lee.shape[3]):
                    data_np[i,j,k,l] = data_lee[i,j,k,l]
    data_np[:,forward_step:] = 0
    data_np = torch.tensor(data_np, dtype=torch.float32)
    result_data = VRNN(data_np)
    result_data = np.asarray(result_data)
    result_data = torch.reshape(torch.tensor(result_data,dtype=torch.float32),(-1,60))
    #print("result_data shape is" + str(result_data.shape))
    frames = result_data
    result = {}
    result['value'] = []
    result['jacobian'] = []
    for i in range(frames.shape[0]):
        var = []
        mean = []
        for j in range(10):
            temp_var_1 = []
            temp_var_2 = []
            temp_mean = []
            temp_mean.append(frames[i][6 * j])
            temp_mean.append(frames[i][1 + 6 * j])
            temp_var_1.append(frames[i][2 + 6 * j]) # in old code there's abs
            temp_var_1.append(frames[i][4 + 6 * j])
            temp_var_2.append(frames[i][5 + 6 * j])
            temp_var_2.append(frames[i][3 + 6 * j])
            temp_var = [temp_var_1, temp_var_2]
            var.append(temp_var)
            mean.append(temp_mean)
        result['value'].append(mean)
        result['jacobian'].append(var)

    result['value'] = np.asarray(result['value'])
    result['value'] = np.reshape(result['value'], (1, int(frame_num), 10, 2))
    result['value'] = torch.from_numpy(result['value'])
    result['jacobian'] = np.asarray(result['jacobian'])
    result['jacobian'] = np.reshape(result['jacobian'], (1, int(frame_num), 10, 2, 2))
    result['jacobian'] = torch.from_numpy(result['jacobian'])
    #for i in range(32):
    #    for j in range(10):
    #        for k in range(2):
    #            for l in range(2):
    #                result['var'][0,i,j,k,l] = kp_data['var'][0,i,j,k,l]
    result['jacobian'] = torch.as_tensor(result['jacobian'],dtype=torch.float32)

    '''
    # debug difference for each dim
    dim1 = torch.mean(torch.abs(result['mean'][:,:,:,0].cpu() - kp_data['mean'][:,:,:,0].cpu()))
    dim2 = torch.mean(torch.abs(result['mean'][:,:,:,1].cpu() - kp_data['mean'][:,:,:,1].cpu()))
    dim3 = torch.mean(torch.abs(result['var'][:,:,:,0,0].cpu() - kp_data['var'][:,:,:,0,0].cpu()))
    dim4 = torch.mean(torch.abs(result['var'][:,:,:,0,1].cpu() - kp_data['var'][:,:,:,0,1].cpu()))
    dim5 = torch.mean(torch.abs(result['var'][:,:,:,1,1].cpu() - kp_data['var'][:,:,:,1,1].cpu()))
    '''
    #print(kp_data['value'])
    # debug difference for each frame
    frame_dim = []
    for i in range(frame_num):
        frame_dim.append(torch.mean(torch.mean(torch.abs(result['value'][:,i,:,:].cpu() - kp_data['value'][:,i,:,:].cpu())) + torch.mean(torch.abs(result['jacobian'][:,i,:,:,:].cpu() - kp_data['jacobian'][:,i,:,:,:].cpu()))))
    #return [dim1,dim2,dim3,dim4,dim5], result
    return frame_dim, result

def VRNN_predict_12(kp_data, VRNN):
    #frame_num = 24
    frame_num = 12 # last line was initial code

    #frame_num = 6
    # initialize VAE
    #print(kp_data)
    original_dim = 50
    order = 2
    forward_step = 5
    changed = False
    iterate = 0
    data_all = kp_data
    first_Lee = True
    
    data = []
    for i in range(frame_num):
        curr_data = []
        for j in range(10):
            curr_data.append(kp_data['value'][0][i][j][0])
            curr_data.append(kp_data['value'][0][i][j][1])
            curr_data.append(kp_data['jacobian'][0][i][j][0][0])
            curr_data.append(kp_data['jacobian'][0][i][j][1][1])
            curr_data.append(kp_data['jacobian'][0][i][j][0][1])
            curr_data.append(kp_data['jacobian'][0][i][j][1][0])
        data.append(curr_data)
    data = np.asarray(data)

    data_lee = np.reshape(data, (-1,int(frame_num),10,6))
    data_np = np.zeros((data_lee.shape[0],data_lee.shape[1],data_lee.shape[2],data_lee.shape[3]))
    for i in range(data_lee.shape[0]):
        for j in range(data_lee.shape[1]):
            for k in range(data_lee.shape[2]):
                for l in range(data_lee.shape[3]):
                    data_np[i,j,k,l] = data_lee[i,j,k,l]
    data_np[:,forward_step:] = 0
    data_np = torch.tensor(data_np, dtype=torch.float32)
    result_data = VRNN(data_np)
    result_data = np.asarray(result_data)
    result_data = torch.reshape(torch.tensor(result_data,dtype=torch.float32),(-1,60))
    #print("result_data shape is" + str(result_data.shape))
    frames = result_data
    result = {}
    result['value'] = []
    result['jacobian'] = []
    for i in range(frames.shape[0]):
        var = []
        mean = []
        for j in range(10):
            temp_var_1 = []
            temp_var_2 = []
            temp_mean = []
            temp_mean.append(frames[i][6 * j])
            temp_mean.append(frames[i][1 + 6 * j])
            temp_var_1.append(frames[i][2 + 6 * j]) # in old code there's abs
            temp_var_1.append(frames[i][4 + 6 * j])
            temp_var_2.append(frames[i][5 + 6 * j])
            temp_var_2.append(frames[i][3 + 6 * j])
            temp_var = [temp_var_1, temp_var_2]
            var.append(temp_var)
            mean.append(temp_mean)
        result['value'].append(mean)
        result['jacobian'].append(var)

    result['value'] = np.asarray(result['value'])
    result['value'] = np.reshape(result['value'], (1, int(frame_num), 10, 2))
    result['value'] = torch.from_numpy(result['value'])
    result['jacobian'] = np.asarray(result['jacobian'])
    result['jacobian'] = np.reshape(result['jacobian'], (1, int(frame_num), 10, 2, 2))
    result['jacobian'] = torch.from_numpy(result['jacobian'])
    #for i in range(32):
    #    for j in range(10):
    #        for k in range(2):
    #            for l in range(2):
    #                result['var'][0,i,j,k,l] = kp_data['var'][0,i,j,k,l]
    result['jacobian'] = torch.as_tensor(result['jacobian'],dtype=torch.float32)

    '''
    # debug difference for each dim
    dim1 = torch.mean(torch.abs(result['mean'][:,:,:,0].cpu() - kp_data['mean'][:,:,:,0].cpu()))
    dim2 = torch.mean(torch.abs(result['mean'][:,:,:,1].cpu() - kp_data['mean'][:,:,:,1].cpu()))
    dim3 = torch.mean(torch.abs(result['var'][:,:,:,0,0].cpu() - kp_data['var'][:,:,:,0,0].cpu()))
    dim4 = torch.mean(torch.abs(result['var'][:,:,:,0,1].cpu() - kp_data['var'][:,:,:,0,1].cpu()))
    dim5 = torch.mean(torch.abs(result['var'][:,:,:,1,1].cpu() - kp_data['var'][:,:,:,1,1].cpu()))
    '''
    #print(kp_data['value'])
    # debug difference for each frame
    frame_dim = []
    for i in range(frame_num):
        frame_dim.append(torch.mean(torch.mean(torch.abs(result['value'][:,i,:,:].cpu() - kp_data['value'][:,i,:,:].cpu())) + torch.mean(torch.abs(result['jacobian'][:,i,:,:,:].cpu() - kp_data['jacobian'][:,i,:,:,:].cpu()))))
    #return [dim1,dim2,dim3,dim4,dim5], result
    return frame_dim, result

def VRNN_predict_12_v(kp_data, VRNN):
    #frame_num = 24
    frame_num = 12 # last line was initial code

    #frame_num = 6
    # initialize VAE
    #print(kp_data)
    original_dim = 50
    order = 2
    forward_step = 5
    changed = False
    iterate = 0
    data_all = kp_data
    first_Lee = True
    
    data = []
    for i in range(frame_num):
        curr_data = []
        for j in range(10):
            curr_data.append(kp_data['value'][0][i][j][0])
            curr_data.append(kp_data['value'][0][i][j][1])
        data.append(curr_data)
    data = np.asarray(data)

    data_lee = np.reshape(data, (-1,int(frame_num),10,2))
    data_np = np.zeros((data_lee.shape[0],data_lee.shape[1],data_lee.shape[2],data_lee.shape[3]))
    for i in range(data_lee.shape[0]):
        for j in range(data_lee.shape[1]):
            for k in range(data_lee.shape[2]):
                for l in range(data_lee.shape[3]):
                    data_np[i,j,k,l] = data_lee[i,j,k,l]
    data_np[:,forward_step:] = 0
    data_np = torch.tensor(data_np, dtype=torch.float32)
    result_data = VRNN(data_np)
    result_data = np.asarray(result_data)
    result_data = torch.reshape(torch.tensor(result_data,dtype=torch.float32),(-1,20))
    #print("result_data shape is" + str(result_data.shape))
    frames = result_data
    result = {}
    result['value'] = []
    result['jacobian'] = []
    for i in range(frames.shape[0]):
        var = []
        mean = []
        for j in range(10):
            temp_mean = []
            temp_mean.append(frames[i][2 * j])
            temp_mean.append(frames[i][1 + 2 * j])
            mean.append(temp_mean)
        result['value'].append(mean)

    result['value'] = np.asarray(result['value'])
    result['value'] = np.reshape(result['value'], (1, int(frame_num), 10, 2))
    result['value'] = torch.from_numpy(result['value'])
    #for i in range(32):
    #    for j in range(10):
    #        for k in range(2):
    #            for l in range(2):
    #                result['var'][0,i,j,k,l] = kp_data['var'][0,i,j,k,l]

    '''
    # debug difference for each dim
    dim1 = torch.mean(torch.abs(result['mean'][:,:,:,0].cpu() - kp_data['mean'][:,:,:,0].cpu()))
    dim2 = torch.mean(torch.abs(result['mean'][:,:,:,1].cpu() - kp_data['mean'][:,:,:,1].cpu()))
    dim3 = torch.mean(torch.abs(result['var'][:,:,:,0,0].cpu() - kp_data['var'][:,:,:,0,0].cpu()))
    dim4 = torch.mean(torch.abs(result['var'][:,:,:,0,1].cpu() - kp_data['var'][:,:,:,0,1].cpu()))
    dim5 = torch.mean(torch.abs(result['var'][:,:,:,1,1].cpu() - kp_data['var'][:,:,:,1,1].cpu()))
    '''
    #print(kp_data['value'])
    # debug difference for each frame
    frame_dim = []
    for i in range(frame_num):
        frame_dim.append(torch.mean(torch.mean(torch.abs(result['value'][:,i,:,:].cpu() - kp_data['value'][:,i,:,:].cpu()))))
    #return [dim1,dim2,dim3,dim4,dim5], result
    return frame_dim, result

def VRNN_predict_12_j(kp_data, VRNN):
    #frame_num = 24
    frame_num = 12 # last line was initial code

    #frame_num = 6
    # initialize VAE
    #print(kp_data)
    original_dim = 50
    order = 2
    forward_step = 5
    changed = False
    iterate = 0
    data_all = kp_data
    first_Lee = True
    
    data = []
    for i in range(frame_num):
        curr_data = []
        for j in range(10):
            curr_data.append(kp_data['jacobian'][0][i][j][0][0])
            curr_data.append(kp_data['jacobian'][0][i][j][1][1])
            curr_data.append(kp_data['jacobian'][0][i][j][0][1])
            curr_data.append(kp_data['jacobian'][0][i][j][1][0])
        data.append(curr_data)
    data = np.asarray(data)

    data_lee = np.reshape(data, (-1,int(frame_num),10,4))
    data_np = np.zeros((data_lee.shape[0],data_lee.shape[1],data_lee.shape[2],data_lee.shape[3]))
    for i in range(data_lee.shape[0]):
        for j in range(data_lee.shape[1]):
            for k in range(data_lee.shape[2]):
                for l in range(data_lee.shape[3]):
                    data_np[i,j,k,l] = data_lee[i,j,k,l]
    data_np[:,forward_step:] = 0
    data_np = torch.tensor(data_np, dtype=torch.float32)
    result_data = VRNN(data_np)
    result_data = np.asarray(result_data)
    result_data = torch.reshape(torch.tensor(result_data,dtype=torch.float32),(-1,40))
    #print("result_data shape is" + str(result_data.shape))
    frames = result_data
    result = {}
    result['jacobian'] = []
    for i in range(frames.shape[0]):
        var = []
        for j in range(10):
            temp_var_1 = []
            temp_var_2 = []
            temp_var_1.append(frames[i][ 4 * j]) # in old code there's abs
            temp_var_1.append(frames[i][2 + 4 * j])
            temp_var_2.append(frames[i][3 + 4 * j])
            temp_var_2.append(frames[i][1 + 4 * j])
            temp_var = [temp_var_1, temp_var_2]
            var.append(temp_var)
        result['jacobian'].append(var)

    result['jacobian'] = np.asarray(result['jacobian'])
    result['jacobian'] = np.reshape(result['jacobian'], (1, int(frame_num), 10, 2, 2))
    result['jacobian'] = torch.from_numpy(result['jacobian'])
    #for i in range(32):
    #    for j in range(10):
    #        for k in range(2):
    #            for l in range(2):
    #                result['var'][0,i,j,k,l] = kp_data['var'][0,i,j,k,l]
    result['jacobian'] = torch.as_tensor(result['jacobian'],dtype=torch.float32)

    '''
    # debug difference for each dim
    dim1 = torch.mean(torch.abs(result['mean'][:,:,:,0].cpu() - kp_data['mean'][:,:,:,0].cpu()))
    dim2 = torch.mean(torch.abs(result['mean'][:,:,:,1].cpu() - kp_data['mean'][:,:,:,1].cpu()))
    dim3 = torch.mean(torch.abs(result['var'][:,:,:,0,0].cpu() - kp_data['var'][:,:,:,0,0].cpu()))
    dim4 = torch.mean(torch.abs(result['var'][:,:,:,0,1].cpu() - kp_data['var'][:,:,:,0,1].cpu()))
    dim5 = torch.mean(torch.abs(result['var'][:,:,:,1,1].cpu() - kp_data['var'][:,:,:,1,1].cpu()))
    '''
    #print(kp_data['value'])
    # debug difference for each frame
    frame_dim = []
    for i in range(frame_num):
        frame_dim.append(torch.mean(torch.mean(torch.abs(result['jacobian'][:,i,:,:,:].cpu() - kp_data['jacobian'][:,i,:,:,:].cpu()))))
    #return [dim1,dim2,dim3,dim4,dim5], result
    return frame_dim, result
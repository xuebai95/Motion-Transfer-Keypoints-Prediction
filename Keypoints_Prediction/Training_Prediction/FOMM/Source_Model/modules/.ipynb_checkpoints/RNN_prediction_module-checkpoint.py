from torch import nn
import torch


class PredictionModule(nn.Module):
    """
    RNN for predicting kp movement
    """

    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule, self).__init__()

        input_size = num_kp * (2 + 4 * (kp_variance == 'matrix'))

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)            
        self.linear = nn.Linear(num_features, input_size)

    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], output.shape[-1]), h

    def forward(self, kp_batch):
        #print("kp_batch mean shape is " + kp_batch['mean'].shape)
        bs, d, num_kp, _ = kp_batch['value'].shape
        inputs = [kp_batch['value'].contiguous().view(bs, d, -1)]
        if 'jacobian' in kp_batch:
            inputs.append(kp_batch['jacobian'].contiguous().view(bs, d, -1))

        input = torch.cat(inputs, dim=-1)

        output, h = self.net(input)
        output = output.view(bs, d, num_kp, -1)
        mean = output[:, :, :, :2]
        #mean = torch.tanh(output[:, :, :, :2])
        kp_array = {'value': mean}
        if 'jacobian' in kp_batch:
            var = output[:, :, :, 2:]
            var = var.view(bs, d, num_kp, 2, 2)
            #var = torch.matmul(var.permute(0, 1, 2, 4, 3), var)
            kp_array['jacobian'] = var

        return kp_array
    
    
######### 6 steps forward prediction given 12 steps, only 'value' ###########
class PredictionModule_v_6(nn.Module):
    """
    RNN for predicting kp movement
    """

    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_v_6, self).__init__()

        input_size = num_kp * (2 + 4 * (kp_variance == 'matrix'))

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)            
        self.linear = nn.Linear(num_features, input_size)

    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], -1), h  # Let the last dimension be determined automatically

    def forward(self, kp_batch):
        bs, d, num_kp, _ = kp_batch['value'].shape
        inputs = [kp_batch['value'].contiguous().view(bs, d, -1)]
        if 'jacobian' in kp_batch:
            inputs.append(kp_batch['jacobian'].contiguous().view(bs, d, -1))

        input = torch.cat(inputs, dim=-1)

        # Use the net method to predict 6 time steps forward instead of d time steps
        output, h = self.net(input[:, :6, :])
        output = output.view(bs, 6, num_kp, -1)

        mean = output[:, :, :, :2]
        kp_array = {'value': mean}
        if 'jacobian' in kp_batch:
            var = output[:, :, :, 2:]
            var = var.view(bs, 6, num_kp, 2, 2)
            kp_array['jacobian'] = var

        return kp_array

######### 4 steps forward prediction given 12 steps, only 'value' ###########
class PredictionModule_v_4(nn.Module):
    """
    RNN for predicting kp movement
    """

    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_v_4, self).__init__()

        input_size = num_kp * (2 + 4 * (kp_variance == 'matrix'))

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)            
        self.linear = nn.Linear(num_features, input_size)

    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], -1), h  # Let the last dimension be determined automatically

    def forward(self, kp_batch):
        bs, d, num_kp, _ = kp_batch['value'].shape
        inputs = [kp_batch['value'].contiguous().view(bs, d, -1)]
        if 'jacobian' in kp_batch:
            inputs.append(kp_batch['jacobian'].contiguous().view(bs, d, -1))

        input = torch.cat(inputs, dim=-1)

        # Use the net method to predict 6 time steps forward instead of d time steps
        output, h = self.net(input[:, :4, :])
        output = output.view(bs, 4, num_kp, -1)

        mean = output[:, :, :, :2]
        kp_array = {'value': mean}
        if 'jacobian' in kp_batch:
            var = output[:, :, :, 2:]
            var = var.view(bs, 4, num_kp, 2, 2)
            kp_array['jacobian'] = var

        return kp_array    
    
############# only use 'value' key for input and prediction #################    
class PredictionModule_v(nn.Module):
    """
    RNN for predicting kp movement
    """
    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_v, self).__init__()

        input_size = num_kp * 2

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.linear = nn.Linear(num_features, input_size)

    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], output.shape[-1]), h
        
    def forward(self, kp_batch):
        bs, d, num_kp, _ = kp_batch['value'].shape
        input = kp_batch['value'].contiguous().view(bs, d, -1)
        output, h = self.net(input)
        output = output.view(bs, d, num_kp, -1)
        mean = output[:, :, :, :2]
        #mean = torch.tanh(output[:, :, :, :2])
        kp_array = {'value': mean}
        
        return kp_array
############# only use one coordinates of 'value' key for input and prediction #################    
class PredictionModule_vv(nn.Module):
    """
    RNN for predicting kp movement
    """

    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_vv, self).__init__()

        input_size = num_kp

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.linear = nn.Linear(num_features, input_size)

    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], output.shape[-1]), h
        
    def forward(self, kp_batch):
        bs, d, num_kp = kp_batch['value'].shape
        input = kp_batch['value'].contiguous().view(bs, d, -1)
        output, h = self.net(input)
        output = output.view(bs, d,-1)
        mean = torch.tanh(output[:, :, :])
        kp_array = {'value': mean}
        
        return kp_array
############# only use 'jacobian' key for input and prediction #################
class PredictionModule_j(nn.Module):
    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_j, self).__init__()
        input_size = num_kp * 4  # Only use jacobian data

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.linear = nn.Linear(num_features, input_size)
    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], output.shape[-1]), h
    def forward(self, kp_batch):
        jacobian = kp_batch['jacobian']
        bs, d, num_kp, _, _ = jacobian.shape  # Extract shape of jacobian data
        input = jacobian.view(bs, d, -1)  # Reshape jacobian data
        output, h = self.net(input)
        output = output.view(bs, d, num_kp, -1)
        var = output[..., :4]
        var = var.view(bs, d, num_kp, 2, 2)
        #var = torch.matmul(var.permute(0, 1, 2, 4, 3), var)
        kp_array = {'jacobian': var}  # Only predict jacobian data
        return kp_array    
    
    
############# 12 frames as input, 6 frames as output, only use 'jacobian' key for input and prediction #################
class PredictionModule_j_6(nn.Module):
    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_j_6, self).__init__()
        input_size = num_kp * 4  # Only use jacobian data

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.linear = nn.Linear(num_features, num_kp * 4)  # Output size is num_kp * 4 for 6 forward steps

    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output[:, :6, :]  # Consider only the first 6 time steps for forward prediction
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], 6, output.shape[-1]), h  # Reshape to (bs, 6, num_kp, 4)

    def forward(self, kp_batch):
        jacobian = kp_batch['jacobian']
        bs, d, num_kp, _, _ = jacobian.shape  # Extract shape of jacobian data
        input = jacobian.view(bs, d, -1)  # Reshape jacobian data
        output, h = self.net(input)
        kp_array = {'jacobian': output.view(bs, 6, num_kp, 2, 2)}  # Predict 6 forward steps for jacobian data
        return kp_array


############# one entry of 'jacobian' key for input and prediction #################
class PredictionModule_jj(nn.Module):
    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_jj, self).__init__()
        input_size = num_kp  # Only use jacobian data

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.linear = nn.Linear(num_features, input_size)
    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], output.shape[-1]), h
    def forward(self, kp_batch):
        jacobian = kp_batch['jacobian']
        bs, d, num_kp = jacobian.shape  # Extract shape of jacobian data
        input = jacobian.view(bs, d, -1)[:, :, :num_kp]  # Reshape jacobian data and select the first num_kp columns
        output, h = self.net(input)
        var = output.view(bs, d, num_kp)
        kp_array = {'jacobian': var}  # Only predict jacobian data
        return kp_array
    
############# two entries of 'jacobian' key for input and prediction #################    
class PredictionModule_j_two(nn.Module):
    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_j_two, self).__init__()
        input_size = num_kp * 2  # Only use two entries of the jacobian matrix

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.linear = nn.Linear(num_features, input_size)
        
    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], output.shape[-1]), h
    
    def forward(self, kp_batch):
        jacobian = kp_batch['jacobian']
        bs, d, num_kp,_ = jacobian.shape  # Extract shape of jacobian data
        input = jacobian.view(bs, d, -1, 2)[:, :, :, :2].contiguous().view(bs, d, -1)  # Reshape jacobian data
        output, h = self.net(input)
        output = output.view(bs, d, num_kp, -1)
        var = torch.cat((output, torch.zeros(bs, d, num_kp, 2 - output.shape[-1],
                                             device=output.device)), dim=-1)
        kp_array = {'jacobian': var}  # Only predict two entries of the jacobian matrix
        return kp_array   
############# three entries of 'jacobian' key for input and prediction #################    
class PredictionModule_j_three(nn.Module):
    def __init__(self, num_kp=10, kp_variance=0.01, num_features=1024, num_layers=1, dropout=0.2):
        super(PredictionModule_j_three, self).__init__()
        input_size = num_kp * 3  # Only use jacobian data

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_features, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.linear = nn.Linear(num_features, input_size)
    def net(self, input, h=None):
        output, h = self.rnn(input, h)
        init_shape = output.shape
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.linear(output)
        return output.view(init_shape[0], init_shape[1], output.shape[-1]), h
    
    def forward(self, kp_batch):
        jacobian = kp_batch['jacobian']
        bs, d, num_kp, _ = jacobian.shape  # Extract shape of jacobian data
        input = jacobian.view(bs, d, -1)[:, :, :30].contiguous().view(bs, d, -1)  # Reshape jacobian data
        output, h = self.net(input)
        output = output.view(bs, d, num_kp, -1)
        var = output[..., :4]
        var = var.view(bs, d, num_kp, 3)
        #var = torch.matmul(var.permute(0, 1, 2, 4, 3), var)
        kp_array = {'jacobian': var}  # Only predict jacobian data
        return kp_array       
    
    
def compute_nmse(predicted_data, ground_truth_data):
    # Calculate the squared L2 norm for each dataset
    sq_norm_pred = torch.norm(predicted_data, dim=2) ** 2
    sq_norm_gt = torch.norm(ground_truth_data, dim=2) ** 2

    # Calculate the L2 norm between predicted and ground truth data
    diff_norm = torch.norm(predicted_data - ground_truth_data, dim=2) ** 2

    # Compute the normalized mean squared error for each dataset
    nmse = diff_norm / sq_norm_gt

    # Optionally, you can take the mean NMSE across all samples in each dataset
    mean_nmse = torch.mean(nmse, dim=(0, 1))  # Taking the mean across batch and time dimensions

    return mean_nmse

# Define a function to compute NMSE loss
def compute_nmse_loss(y_true, y_pred):
    numerator = torch.mean((y_true - y_pred)**2)
    denominator = torch.mean(y_true**2)
    nmse_loss = numerator / (denominator + 1e-6)  # Adding a small value to avoid division by zero
    return nmse_loss

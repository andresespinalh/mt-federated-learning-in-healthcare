import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.classification as M
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from model_architectures import FNNEegSignals

from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd
import argparse, json
import warnings
import logging
from IPython import display
from scipy.signal import savgol_filter
import subprocess
import os

# Federated Setting
import flwr as fl
from typing import Dict, List, Tuple
from collections import OrderedDict
from FNNFederatedClient import FederatedClient

def get_vram_usage():
    try:
        cmd = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            vram_used = int(result.stdout.strip())
            return vram_used
        else:
            print("Error:", result.stderr, flush='True')
    except Exception as e:
        print("An error occurred:", e, flush='True')
    return None

### Helpers
def train(model, x_train, y_train, exp_conf, exp_name, global_params, proximal_mu):
    device = torch.device('cuda:0')
    # Store the local parameters before training for FedProx
    local_params = list(model.parameters())
    # Get the value to scale the weights based on class imbalance
    n_pos = torch.tensor(y_train[y_train==1].shape[0]).float()
    n_neg = torch.tensor(y_train[y_train==0].shape[0]).float()
    pos_weight = n_neg / n_pos

    # Loss Function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = getattr(torch.optim, exp_conf['optimizer'])
    optimizer = optimizer(model.parameters(), lr=exp_conf['learning_rate'])

    # Metrics Accumulators
    train_losses = []
    train_accuracies = []
    train_sensitivities = []
    train_specificities = []

    # Define evaluation metrics settings
    accuracy_metric = M.BinaryAccuracy(threshold=exp_conf['decision_boundary']).to(device)
    recall_metric = M.BinaryRecall(threshold=exp_conf['decision_boundary']).to(device)
    specificity_metric = M.BinarySpecificity(threshold=exp_conf['decision_boundary']).to(device)

    # For each training epoch
    for epoch_i in range(exp_conf['n_epochs']):
        ## Training
        model.train()
        
        # Update on progress
        # print(f'Running {exp_name}, epoch {epoch_i} of {exp_conf["n_epochs"]-1}', flush=True)
        
        # Forward Pass
        y_hat = model(x_train)

        # Compute the Loss
        train_loss = loss_function(y_hat, y_train)

        # Add regularization term to the loss function if the Federated Strategy is 'FedProx'
        if (exp_conf['federated_strategy'] == 'FedProx') & (global_params is not None):
            device = torch.device('cuda:0')
            proximal_term = torch.tensor(0).float().to(device)

            # Calculate the proximal term (Difference between local and global weights)
            for local_weights, global_weights in zip(local_params, global_params):
                local_weights = torch.tensor(local_weights).float().to(device)
                global_weights = torch.tensor(global_weights).float().to(device)
                proximal_term += (local_weights - global_weights).norm(2)
                
            train_loss += (proximal_mu / 2) * proximal_term

        train_losses.append(train_loss)

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        ## Compute epoch training metrics
        y_pred = torch.sigmoid(y_hat)
        train_accuracy = accuracy_metric(y_pred, y_train)
        train_accuracies.append(train_accuracy)
        train_sensitivity = recall_metric(y_pred, y_train)
        train_sensitivities.append(train_sensitivity)
        train_specificity = specificity_metric(y_pred, y_train)
        train_specificities.append(train_specificity)


        vram_used = get_vram_usage()
        dict_memory = {
            'federated_strategy': [exp_conf['federated_strategy']]
            , 'client_config': [client_config]
            , 'client_id': [client_id]
            , 'epoch': [epoch_i]
            , 'VRAM': [vram_used]
        }

        # print(dict_memory, flush=True
        df_memory = pd.DataFrame.from_dict(dict_memory)
        filename = f'{exp_conf["logs_path"]}/fnn_memory.csv'
        df_memory.to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))

    # Collect all objects in CPU
    result = {
        # Training
        'train_losses': torch.as_tensor(train_losses).cpu()
        , 'train_accuracies': torch.as_tensor(train_accuracies).cpu()
        , 'train_sensitivities': torch.as_tensor(train_sensitivities).cpu()
        , 'train_specificities': torch.as_tensor(train_specificities).cpu()
    }

    return result

### Run this Federated Client
warnings.filterwarnings("ignore")

# Get the client configuration and ID from the terminal
parser = argparse.ArgumentParser()
parser.add_argument('client_config', help='Federated Client Configuration (Number of FL Clients for this execution)')
parser.add_argument('client_id', help='ID of this Federated Client')
args = parser.parse_args()
client_config = args.client_config
client_id = args.client_id

# Fetch experiment configuration from disk
with open('exp_config_federated.json', 'r') as file:
    json_data = file.read()
    exp_conf = json.loads(json_data)

# Change this to swap between CPU/GPU
device = torch.device('cuda:0')
# device = torch.device('cpu')

### Read and normalize the data for this client
exp_conf['train_path'] = f'{exp_conf["data_path"]}/fl-{exp_conf["dataset"]}'

x_train = pd.read_csv(f"{exp_conf['train_path']}/{client_config}-flclients/eeg_x_train_flc{client_id}.csv")
y_train = pd.read_csv(f"{exp_conf['train_path']}/{client_config}-flclients/eeg_y_train_flc{client_id}.csv")



# Normalization (Z-Score)
# Training Set
x_train_mean = x_train.mean()
x_train_sd = x_train.std() 
x_train_norm = (x_train-x_train_mean)/x_train_sd
x_train = x_train_norm

# Load model and data
fnn = FNNEegSignals(exp_conf['n_layers'], exp_conf['n_units'], exp_conf['perc_dropout'])
fnn.to(device)

# Load the data sets into PyTorch Tensors
x_train = torch.tensor(x_train.values, device=device).float()
y_train = torch.tensor(y_train.values, device=device).float()

data = {'x_train': x_train, 'y_train': y_train}

# Start client
logging.getLogger('flwr').setLevel(logging.INFO)
fl_client = FederatedClient(fnn, data, train)
fl.client.start_numpy_client(server_address="localhost:5040", client=fl_client)
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import sys, os, json, time, cv2, shutil, argparse
from CaraNetFederatedClient import FederatedClient
import warnings
import logging
import subprocess

from torch.autograd import Variable
from datetime import datetime
from collections import OrderedDict
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from CaraNet import caranet

# Federated Setting
import flwr as fl
from typing import Dict, List, Tuple
from collections import OrderedDict

## Helpers
# Calculate the loss
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()

# Trains the model for one epoch
def train_epoch(train_loader, model, optimizer, epoch, exp_config, total_step, global_params, proximal_mu):
    # Store the local parameters before training for FedProx
    local_params = list(model.parameters())
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    epoch_loss_record = AvgMeter()
    train_mdice_record = AvgMeter()
    
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(exp_config['trainsize']*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss1 = structure_loss(lateral_map_1, gts)
            
            # Structure Loss
            loss = loss5 + loss3 + loss2 + loss1

            # Add regularization term to the loss function if the Federated Strategy is 'FedProx'
            if (exp_config['federated_strategy'] == 'FedProx') & (global_params is not None):
                device = torch.device('cuda:0')
                proximal_term = torch.tensor(0).float().to(device)

                # Calculate the proximal term (Difference between local and global weights)
                for local_weights, global_weights in zip(local_params, global_params):
                    local_weights = torch.tensor(local_weights).float().to(device)
                    global_weights = torch.tensor(global_weights).float().to(device)
                    proximal_term += (local_weights - global_weights).norm(2)
                    
                loss += (proximal_mu / 2) * proximal_term
            
            ## Dice Metric
            batch_dice = []

            # For each ground truth mask in this batch
            for gt_id in range(0, len(gts)):
                gt = gts[gt_id].cpu()

                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)

                res = lateral_map_5[gt_id]
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                
                input = res
                target = np.array(gt)
                N = gt.shape
                smooth = 1
                input_flat = np.reshape(input,(-1))
                target_flat = np.reshape(target,(-1))
        
                intersection = (input_flat*target_flat)
                
                # Calculate the image dice metric and append it to the batch metrics
                img_dice =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
                batch_dice.append(img_dice)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, exp_config['clip'])
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record5.update(loss5.data, exp_config['batchsize'])
                loss_record3.update(loss3.data, exp_config['batchsize'])
                loss_record2.update(loss2.data, exp_config['batchsize'])
                loss_record1.update(loss1.data, exp_config['batchsize'])
                epoch_loss_record.update(loss.detach().cpu(), exp_config['batchsize'])
                train_mdice_record.update(torch.tensor(batch_dice).mean(), exp_config['batchsize'])
    
    train_results_dict = {
        'train_loss': float(epoch_loss_record.show().numpy())
        , 'train_mdice': float(train_mdice_record.show().numpy())
        , 'epoch': epoch
        , 'exp_name': exp_config['exp_name']
    }


    return train_results_dict

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

# Train the model
def train(train_loader, model, optimizer, exp_config, client_id, global_params, proximal_mu):
    train_loss = []
    train_mdice = []
    total_step = len(train_loader)

    for epoch in range(1, exp_config['epoch'] + 1):
        adjust_lr(optimizer, exp_config['lr'], epoch, 0.1, 200)
        res_epoch = train_epoch(train_loader, model, optimizer, epoch, exp_config, total_step, global_params, proximal_mu)
        train_loss.append(res_epoch['train_loss'])
        train_mdice.append(res_epoch['train_mdice'])
        print(f'Client {client_id} Epoch [{epoch}/{exp_config["epoch"]}] Done', flush=True)

        vram_used = get_vram_usage()
        dict_memory = {
            'federated_strategy': [exp_config['federated_strategy']]
            , 'client_config': [client_config]
            , 'client_id': [client_id]
            , 'epoch': [epoch]
            , 'VRAM': [vram_used]
        }

        # print(dict_memory, flush=True
        df_memory = pd.DataFrame.from_dict(dict_memory)
        filename = f'{exp_conf["logs_path"]}/fnn_memory.csv'
        df_memory.to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))

    return {'train_loss': np.array(train_loss), 'train_mdice': np.array(train_mdice)}

## Run this Federated Client
warnings.filterwarnings("ignore")

# Get the client configuration and ID from the terminal
parser = argparse.ArgumentParser()
parser.add_argument('client_config', help='Federated Client Configuration (Number of FL Clients for this execution)')
parser.add_argument('client_id', help='ID of this Federated Client')
parser.add_argument('batch_size', help='Batch size for this client')
args = parser.parse_args()
client_config = args.client_config
client_id = args.client_id
batch_size = args.batch_size

# Fetch experiment configuration from disk
with open('exp_config_federated.json', 'r') as file:
    json_data = file.read()
    exp_conf = json.loads(json_data)

exp_conf['batchsize'] = int(batch_size)

# Path for the data of this client
train_path = f'../../../data/inputs/kvasir_federated/{client_config}_flclients/flclient_{client_id}'

device = torch.device('cuda:0')

## Load model and data
model = caranet().to(device)

# Load Data
image_root = '{}/images/'.format(train_path)
gt_root = '{}/masks/'.format(train_path)

train_loader = get_loader(image_root, gt_root, batchsize=exp_conf['batchsize'], trainsize=exp_conf['trainsize'], augmentation = True)

# Start Client
logging.getLogger('flwr').setLevel(logging.INFO)
fl_client = FederatedClient(model, train_loader, client_id, train)
fl.client.start_numpy_client(server_address="localhost:5040", client=fl_client)

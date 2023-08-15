import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os, json, time, cv2, shutil, argparse, copy

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

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, client_id, train_func) -> None:
        self.model = model
        self.train_loader = train_loader
        self.train_func = train_func
        self.client_id = client_id

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        n_examples = len(self.train_loader.dataset)

        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), config['lr'])
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), config['lr'], weight_decay = 1e-4, momentum = 0.9)


        # Do for FedProx strategy
        if config['federated_strategy'] == 'FedProx':
            global_params = copy.deepcopy(list(self.model.parameters()))

            result_train = self.train_func(
                self.train_loader, self.model, optimizer, config, self.client_id, global_params, config['proximal_mu']
            )
        else:
            result_train = self.train_func(
                self.train_loader, self.model, optimizer, config, self.client_id, None, None
            )

        metrics = {
            'train_loss': result_train['train_loss'].mean()
            , 'train_mdice': result_train['train_mdice'].mean()
        }

        return self.get_parameters(config={}), n_examples, metrics
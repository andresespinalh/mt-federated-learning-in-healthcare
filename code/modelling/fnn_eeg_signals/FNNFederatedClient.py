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
import matplotlib.pyplot as plt
import gc
import time
import os
import copy
from IPython import display
from scipy.signal import savgol_filter
display.set_matplotlib_formats('svg')

# Federated Setting
import flwr as fl
from typing import Dict, List, Tuple
from collections import OrderedDict

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, data, train_func) -> None:
        self.model = model
        self.data = data
        self.train_func = train_func

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        n_examples = self.data['x_train'].shape[0]

        if config['federated_strategy'] == 'FedProx':
            global_params = copy.deepcopy(list(self.model.parameters()))
            result_train = self.train_func(
                self.model, self.data['x_train'], self.data['y_train']
                , config, config['exp_name'], global_params, config['proximal_mu']
            )
        else:
            result_train = self.train_func(
                self.model, self.data['x_train'], self.data['y_train']
                , config, config['exp_name'], None, None
            )

        metrics = {
            'loss': result_train['train_losses'].mean().item()
            , 'accuracy': result_train['train_accuracies'].mean().item()
            , 'sensitivity': result_train['train_sensitivities'].mean().item()
            , 'specificity': result_train['train_specificities'].mean().item()
        }

        return self.get_parameters(config={}), n_examples, metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defines a modular FNN model architecture for the EEG Signals data
class FNNEegSignals(nn.Module):
    def __init__(self, n_layers, n_units, perc_dropout): 
        super().__init__()

        # Set Attributes
        self.n_layers = n_layers
        self.n_units = n_units
        self.perc_droput = perc_dropout

        ## Layers
        # Dictionary to store the layers
        self.layers = nn.ModuleDict()

        # Input Layer
        self.layers['input'] = nn.Linear(53, n_units)

        # Hidden Layers
        for i in range(n_layers):
            self.layers[f'hidden_{i}'] = nn.Linear(n_units, n_units)
        
        # Output Layer
        self.layers['output'] = nn.Linear(n_units, 1)
    
    def forward(self, x):
        # Input Layer
        x = F.relu(self.layers['input'](x))
        x = F.dropout(x, p=self.perc_droput, training=self.training) # Dropout for the input layer

        # Hidden Layers
        for i in range(self.n_layers):
            curr_hidden_layer = self.layers[f'hidden_{i}']
            x = F.relu(curr_hidden_layer(x))
            x = F.dropout(x, p=self.perc_droput, training=self.training) # Dropout for the hidden layers
        
        # Output Layer
        x = self.layers['output'](x)

        return x
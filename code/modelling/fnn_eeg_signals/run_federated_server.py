import flwr as fl
import torch
import torch.nn as nn
import torchmetrics.classification as M
import json
import warnings

from flwr.common import Metrics, NDArrays, Scalar, Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Tuple, Optional, OrderedDict, Union
from model_architectures import FNNEegSignals

import os, gc, time, argparse, shutil
import logging
import pandas as pd
import numpy as np

### Helper Functions
# Evaluate the model (Used for Centralized FL Evaluation)
def evaluate(model, x_val, y_val, exp_conf):
    model.eval()

    # Get the value to scale the weights based on class imbalance
    n_pos = torch.tensor(y_val[y_val==1].shape[0]).float()
    n_neg = torch.tensor(y_val[y_val==0].shape[0]).float()
    pos_weight = n_neg / n_pos

    # Loss Function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        y_hat = model(x_val)
        loss = loss_function(y_hat, y_val)
        y_pred = torch.sigmoid(y_hat)

        # Define evaluation metrics settings
        accuracy_metric = M.BinaryAccuracy(threshold=exp_conf['decision_boundary']).to(device)
        recall_metric = M.BinaryRecall(threshold=exp_conf['decision_boundary']).to(device)
        specificity_metric = M.BinarySpecificity(threshold=exp_conf['decision_boundary']).to(device)

        # Compute metrics
        accuracy = accuracy_metric(y_pred, y_val)
        sensitivity = recall_metric(y_pred, y_val)
        specificity = specificity_metric(y_pred, y_val)

        results = {'loss': loss, 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity}

    return results

# Test the aggregated FL model
def test(model, x_val, y_val, exp_conf):
    model.eval()

    # Get the value to scale the weights based on class imbalance
    n_pos = torch.tensor(y_val[y_val==1].shape[0]).float()
    n_neg = torch.tensor(y_val[y_val==0].shape[0]).float()
    pos_weight = n_neg / n_pos

    # Loss Function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        y_hat = model(x_val)
        loss = loss_function(y_hat, y_val)
        y_pred = torch.sigmoid(y_hat)

        # Define test metrics settings
        accuracy_metric = M.BinaryAccuracy(threshold=exp_conf['decision_boundary']).to(device)
        recall_metric = M.BinaryRecall(threshold=exp_conf['decision_boundary']).to(device)
        specificity_metric = M.BinarySpecificity(threshold=exp_conf['decision_boundary']).to(device)
        confusion_matrix_metric = M.BinaryConfusionMatrix(threshold=exp_conf['decision_boundary']).to(device)
        roc_metric = M.BinaryROC(thresholds=100).to(device)
        auroc_metric = M.BinaryAUROC().to(device)

        # Compute metrics
        accuracy = accuracy_metric(y_pred, y_val)
        sensitivity = recall_metric(y_pred, y_val)
        specificity = specificity_metric(y_pred, y_val)
        confusion_matrix = confusion_matrix_metric(y_pred, y_val)
        auroc = auroc_metric(y_pred, y_val)
        roc_fpr, roc_tpr, roc_thresholds = roc_metric(y_pred, y_val.long())

        results = {
            'loss': loss, 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity
            , 'confusion_matrix': confusion_matrix, 'roc_fpr': roc_fpr.cpu(), 'roc_tpr': roc_tpr.cpu()
            , 'roc_thresholds': roc_thresholds.cpu(), 'auroc': auroc
        }

    return results

# Evaluate the model each FL round using the test set
def get_evaluate_fn(model, x_val, y_val, exp_conf):
    # The `evaluate` function will be called after every round
    def evaluate_fn(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        results_test = evaluate(model, x_val, y_val, exp_conf)
        
        result = {
            'loss': results_test['loss'].item(), 'accuracy': results_test['accuracy'].item()
            , 'sensitivity': results_test['sensitivity'].item(), 'specificity': results_test['specificity'].item()}
        
        return results_test['loss'], result

    return evaluate_fn

# Aggregate training metrics for Federated Evaluation
def agg_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Collect all the FL Client metrics and weight them
    losses = [n_examples * metric['loss'] for n_examples, metric in metrics]
    accuracies = [n_examples * metric['accuracy'] for n_examples, metric in metrics]
    specificities = [n_examples * metric['specificity'] for n_examples, metric in metrics]
    sensitivities = [n_examples * metric['sensitivity'] for n_examples, metric in metrics]

    total_examples = sum([n_examples for n_examples, _ in metrics])

    # Compute weighted averages
    agg_metrics = {
        'loss': sum(losses) / total_examples
        , 'accuracy': sum(accuracies) / total_examples
        , 'specificity': sum(specificities) / total_examples
        , 'sensitivity': sum(sensitivities) / total_examples
    }

    return agg_metrics

def compile_results(rounds_results, inference_results, exp_conf, best_model_round, elapsed_time, filepath, loss_tuning):
    ## Training Results (Wide Format)
    results_wide_df = pd.DataFrame.from_dict(rounds_results, orient='columns')
    results_wide_df['round'] = [round + 1 for round in range(exp_conf['n_rounds'])]
    results_wide_df['exp_name'] = exp_conf['exp_name']

    ## Training Results (Long Format)
    results_long_df = pd.melt(frame=results_wide_df, id_vars=['round', 'exp_name'], value_vars=rounds_results.keys(), var_name='metric', value_name='value')
    results_long_df = results_long_df.sort_values('round')

    ## Training Summary 
    roc_dict = {key: inference_results[key].tolist() for key in ['roc_fpr', 'roc_tpr', 'roc_thresholds']}

    exp_conf_simpl = {key: value for key, value in exp_conf.items() if key not in ['data_path', 'models_path', 'logs_path']}

    summary_exp = {
        'exp_name': [exp_conf['exp_name']]
        , 'exp_type': [exp_conf['experiment_type']]
        , 'exp_resource': [exp_conf['resource']]
        , 'exp_device': [exp_conf['device']]
        # , 'exp_k_folds': [exp_conf['k-folds']]
        , 'exp_configuration': [json.dumps(exp_conf_simpl)]
        , 'elapsed_time': [elapsed_time]
        , 'round_best_model': best_model_round
        , 'loss': [inference_results['loss'].item()]
        , 'accuracy': [inference_results['accuracy'].item()]
        , 'sensitivity': [inference_results['sensitivity'].item()]
        , 'specificity': [inference_results['specificity'].item()]
        , 'auroc': [inference_results['auroc'].item()]
        , 'tp': [inference_results['confusion_matrix'].flatten()[0].item()]
        , 'fn': [inference_results['confusion_matrix'].flatten()[1].item()]
        , 'fp': [inference_results['confusion_matrix'].flatten()[2].item()]
        , 'tn': [inference_results['confusion_matrix'].flatten()[3].item()]
        # , 'roc_thresholds': [json.dumps(roc_dict)]
        , 'loss_tuning': [loss_tuning]
    }

    exp_summary_df = pd.DataFrame.from_dict(summary_exp, orient='columns')

    # Persist experiments to disk (If path is provided)
    if(filepath!=None):
        filename_train_wide = f'{filepath}/results_test_wide_federated.csv'
        results_wide_df.to_csv(filename_train_wide, index=False, mode='a', header=not os.path.exists(filename_train_wide))
        filename_train_long = f'{filepath}/results_test_long_federated.csv'
        results_long_df.to_csv(filename_train_long, index=False, mode='a', header=not os.path.exists(filename_train_long))
        filename_summary = f'{filepath}/exp_summary_test_federated.csv'
        exp_summary_df.to_csv(filename_summary, index=False, mode='a', header=not os.path.exists(filename_summary))
    
    return {
        'results': {'wide': results_wide_df, 'long': results_long_df}
        , 'summary_exp': exp_summary_df
    }

# Wrapper to send experiment configuration to FL clients
def get_fit_config_fn(exp_conf):
    def fit_config(server_round: int):
        return exp_conf
    
    return fit_config

### Experiment Configuration
warnings.filterwarnings("ignore")

## Terminal Arguments
parser = argparse.ArgumentParser()
parser.add_argument('federated_strategy', help='Federated Algorithm for this run')
parser.add_argument('n_clients', help='Client configuration for this run')
parser.add_argument('param_set', help='Additional Parameters')
args = parser.parse_args()

federated_params = json.loads(str(args.param_set))

# Fetch experiment configuration from disk
with open('exp_config_federated.json', 'r') as file:
    json_data = file.read()
    exp_conf = json.loads(json_data)

## Update experiment configurations
exp_conf['federated_strategy'] = args.federated_strategy
exp_conf['n_clients'] = int(args.n_clients)
exp_conf['train_path'] = f'{exp_conf["data_path"]}/fl-{exp_conf["dataset"]}'
exp_conf['test_path'] = f'{exp_conf["data_path"]}/centralized-{exp_conf["dataset"]}'

# Update the experiment name
str_params = ''
if(federated_params is not None):
    # Add the parameters (If any) to the name of the experiment
    for param in federated_params:
        str_params += f'_{param}:{federated_params[param]}'

    # Add strategy specific parameters to the experiment configuration
    for parameter in federated_params.keys():
        exp_conf[parameter] = federated_params[parameter]

exp_conf['exp_name'] = f'{exp_conf["exp_name"]}_{exp_conf["dataset"]}_{exp_conf["federated_strategy"]}_{exp_conf["n_clients"]}FLC{str_params}'

print(f'Experiment Name: "{exp_conf["exp_name"]}"...', flush=True)
print(f'Experiment Configuration: {exp_conf}\n', flush=True)

# Use GPU
device = torch.device('cuda:0')

# Clean-Up Models Folder before run
if os.path.exists(exp_conf['models_path']):
    shutil.rmtree(exp_conf['models_path'])

os.makedirs(exp_conf['models_path'])

# Custom implementation of the selected federated strategy
class FedCustom(getattr(fl.server.strategy, exp_conf['federated_strategy'])):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            filepath_model = f'{exp_conf["models_path"]}/round-{server_round}-weights.npz'
            np.savez(filepath_model, *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

### Read and pre-process the Data
x_val = pd.read_csv(f"{exp_conf['test_path']}/eeg_x_val.csv")
y_val = pd.read_csv(f"{exp_conf['test_path']}/eeg_y_val.csv")
x_test = pd.read_csv(f"{exp_conf['test_path']}/eeg_x_test.csv")
y_test = pd.read_csv(f"{exp_conf['test_path']}/eeg_y_test.csv")

# Normalization (Z-Score)
# Validation Set
x_val_mean = x_val.mean()
x_val_sd = x_val.std() 
x_val_norm = (x_val-x_val_mean)/x_val_sd
x_val = x_val_norm

# Test Set
x_test_mean = x_test.mean()
x_test_sd = x_test.std() 
x_test_norm = (x_test-x_test_mean)/x_test_sd
x_test = x_test_norm

### Create and run the model
# Move data to device
x_val = torch.tensor(x_test.values, device=device).float()
y_val = torch.tensor(y_test.values, device=device).float()
x_test = torch.tensor(x_test.values, device=device).float()
y_test = torch.tensor(y_test.values, device=device).float()

# Create the model
fnn = FNNEegSignals(exp_conf['n_layers'], exp_conf['n_units'], exp_conf['perc_dropout'])
fnn.to(device)

# Get the first round of parameters from the server
parameters = [val.cpu().numpy() for _, val in fnn.state_dict().items()]

min_clients = exp_conf['n_clients']

# Default parameters for the most basic FL Strategy (FedAvg)
strategy_params_dict = {
    'fit_metrics_aggregation_fn': agg_metrics
    , 'evaluate_fn': get_evaluate_fn(fnn, x_val, y_val, exp_conf)
    , 'fraction_evaluate': 0
    , 'min_available_clients': min_clients
    , 'min_fit_clients': min_clients
    , 'fraction_fit': 1.0
    , 'on_fit_config_fn': get_fit_config_fn(exp_conf)
    , 'initial_parameters' : fl.common.ndarrays_to_parameters(parameters)
}

# Update Parameters for the other Federated Strategies (FedAdam, FedAdagrad, FedYogi, FedProx)
if(exp_conf['federated_strategy']=='FedProx'):
    strategy_params_dict['proximal_mu'] = float(exp_conf['prox_mu'])
elif(exp_conf['federated_strategy'] in ['FedAdam', 'FedYogi']):
    strategy_params_dict['beta_1'] = 0.9
    strategy_params_dict['beta_2'] = 0.99
    strategy_params_dict['tau'] = float(exp_conf['tau'])
    strategy_params_dict['eta'] = float(exp_conf['eta'])
    strategy_params_dict['eta_l'] = float(exp_conf['eta_l'])
elif(exp_conf['federated_strategy'] == 'FedAdagrad'):
    strategy_params_dict['tau'] = float(exp_conf['tau'])
    strategy_params_dict['eta'] = float(exp_conf['eta'])
    strategy_params_dict['eta_l'] = float(exp_conf['eta_l'])

fl_strategy = FedCustom(**strategy_params_dict)
    
start_time = time.time()
# Start the FL Server
logging.getLogger('flwr').setLevel(logging.INFO)
result = fl.server.start_server(server_address="localhost:5040", strategy=fl_strategy, config=fl.server.ServerConfig(num_rounds=exp_conf['n_rounds']))
end_time = time.time()
elapsed_time = end_time - start_time

# Gather results and output to disk
print(f'\n===FL Experiment {exp_conf["exp_name"]} done, performing Federated Testing===', flush=True)
### Final Model Testing
# Format results
rounds_train_results = {key: np.array([metric for _, metric in result.metrics_distributed_fit[key]]) for key in result.metrics_distributed_fit.keys()} # Federated
rounds_test_results = {key: np.array([metric for _, metric in result.metrics_centralized[key]]) for key in result.metrics_centralized.keys()} # Centralized
rounds_test_results = {key: value[1:] for key, value in rounds_test_results.items()} 

last_n_rounds = 125
last_n_rounds_vals = rounds_test_results['loss'][-last_n_rounds:]
loss_tuning = np.array([last_n_rounds_vals]).mean()

## Retrieve best model and load it for inference
# Note: argmin() returns a 0-based index and there are server_rounds + 1 results from the FL run
best_model_round = rounds_test_results['loss'].argmin() + 1

# Load stored weights for best model
filepath_model = f'{exp_conf["models_path"]}/round-{best_model_round}-weights.npz'

parameters = []
with np.load(filepath_model) as weights_best_model:
    for parameter in weights_best_model:
        parameters.append(weights_best_model[parameter])

# Create the model
model = FNNEegSignals(exp_conf['n_layers'], exp_conf['n_units'], exp_conf['perc_dropout'])
model.to(device)

# Create the state_dict based of the retrieved weights
params_dict = zip(model.state_dict().keys(), parameters)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

## Save the model state_dict
items_path_str = exp_conf['models_path'].split('/')
filepath_fed_model = ''

# Move up one directory to save the best model
for i, item in enumerate(items_path_str):
    if(i!=len(items_path_str)-1):
        filepath_fed_model += item + '/'

filepath_fed_model += f'{exp_conf["exp_name"]}_Federated.pt'

if (filepath_fed_model!=None):
    torch.save(state_dict, filepath_fed_model)

# Set the model parameters 
model.load_state_dict(state_dict, strict=True)

# Perform inference with the stored model
inference_results = test(model, x_test, y_test, exp_conf)

# Clean staging directory after retrieving the best model
filepath_stage = exp_conf['models_path']
for file in os.listdir(filepath_stage):
    os.remove(f'{filepath_stage}/{file}')

### Persist Results to Disk
# Compile train/test results and format before persisting them
mapping_train = {'loss': 'train_losses', 'accuracy': 'train_accuracies', 'specificity': 'train_specificities', 'sensitivity': 'train_sensitivities'}
mapping_test = {'loss': 'test_losses', 'accuracy': 'test_accuracies', 'specificity': 'test_specificities', 'sensitivity': 'test_sensitivities'}
rounds_train_results_f = {mapping_train[key]: rounds_train_results[key] for key in rounds_train_results}
# Discard the first test result (From the random client initialization) as it is not a result of FedAvg
rounds_test_results_f = {mapping_test[key]: rounds_test_results[key] for key in rounds_train_results} 
rounds_results = {**rounds_train_results_f, **rounds_test_results_f}

# Persist Results to Disk if a filepath is provided
filepath = exp_conf['logs_path']
# filepath = None
res = compile_results(rounds_results, inference_results, exp_conf, best_model_round, elapsed_time, filepath, loss_tuning)

print(f'===FL Experiment {exp_conf["exp_name"]} concluded===\n', flush=True)
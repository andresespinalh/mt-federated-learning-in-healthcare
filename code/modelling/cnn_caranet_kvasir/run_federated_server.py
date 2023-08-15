import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os, json, time, cv2, shutil, argparse
import warnings, logging

from torch.autograd import Variable
from datetime import datetime
from collections import OrderedDict
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from CaraNet import caranet

# Federated Setting
from typing import Dict, List, Tuple
from collections import OrderedDict
import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Tuple, Optional, OrderedDict, Union

### Helpers
# Evaluate the model (Used for Centralized FL Evaluation)
def evaluate(model, data_path):
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    b=0.0

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res3, res2, res1 = model(image)

        # Dice
        res = res5
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))

        intersection = (input_flat*target_flat)
        # print(f'Raw res5 value is {res5.sum()}')

        dice =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(dice)
        a = float(a)
        b = b + a

    mdice = b/test_loader.size

    # Fixed, this should vary according to the test set size (Rather than be fixed)
    return mdice

# Evaluate the model each FL round using the test set
def get_evaluate_fn(model, data_path):
    # The `evaluate` function will be called after every round
    def evaluate_fn(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        results_test = evaluate(model, data_path)
        return 1, {'evaluate_mdice': results_test}
    return evaluate_fn

# Aggregate training metrics for Federated Evaluation
def agg_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Collect all the FL Client metrics and weight them
    train_losses = [n_examples * metric['train_loss'] for n_examples, metric in metrics]
    train_mdices = [n_examples * metric['train_mdice'] for n_examples, metric in metrics]

    total_examples = sum([n_examples for n_examples, _ in metrics])

    # Compute weighted averages
    agg_metrics = {
        'train_loss': sum(train_losses) / total_examples
        , 'train_mdice': sum(train_mdices) / total_examples
    }

    return agg_metrics

# Summarize results and dump to disk
def compile_results(rounds_results, inference_results, exp_conf, best_model_round, elapsed_time, filepath):
    ## Training Results (Wide Format)
    results_wide_df = pd.DataFrame.from_dict(rounds_results, orient='columns')
    results_wide_df['round'] = [round + 1 for round in range(exp_conf['n_rounds'])]
    results_wide_df['exp_name'] = exp_conf['exp_name']

    ## Training Results (Long Format)
    results_long_df = pd.melt(frame=results_wide_df, id_vars=['round', 'exp_name'], value_vars=rounds_results.keys(), var_name='metric', value_name='value')
    results_long_df = results_long_df.sort_values('round')

    ## Training Summary
    exp_conf_simpl = {key:value for key, value in exp_conf.items() if key not in ['test_path', 'evaluate_path', 'models_path', 'logs_path']}

    summary_exp = {
        'exp_name': [exp_conf['exp_name']]
        , 'exp_type': [exp_conf['experiment_type']]
        , 'exp_resource': [exp_conf['resource']]
        , 'exp_device': [exp_conf['device']]
        , 'exp_configuration': [json.dumps(exp_conf_simpl)]
        , 'elapsed_time': [elapsed_time]
        , 'round_best_model': best_model_round
        , 'mdice': [inference_results]
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
parser.add_argument('evaluate_client', help='Client ID for K-Folds CV evaluation')
parser.add_argument('param_set', help='Additional Parameters')
parser.add_argument('batch_size', help='Batch size for this client')
args = parser.parse_args()

federated_params = json.loads(str(args.param_set))

# Fetch experiment configuration from disk
with open('exp_config_federated.json', 'r') as file:
    json_data = file.read()
    exp_conf = json.loads(json_data)

## Update experiment configuration with command line arguments
exp_conf['federated_strategy'] = args.federated_strategy
exp_conf['n_clients'] = int(args.n_clients)
exp_conf['evaluate_client'] = int(args.evaluate_client)
exp_conf['batchsize'] = int(args.batch_size)

# Update the experiment name
# Add the parameters (If any) to the name
str_params = ''
if(federated_params is not None):
    # Add the parameters (If any) to the name of the experiment
    for param in federated_params:
        str_params += f'_{param}:{federated_params[param]}'

    # Add strategy specific parameters to the experiment configuration
    for parameter in federated_params.keys():
        exp_conf[parameter] = federated_params[parameter]

exp_conf['exp_name'] = f'{exp_conf["exp_name"]}_{exp_conf["federated_strategy"]}_{exp_conf["n_clients"]}FLC{str_params}'

print(f'Experiment Name: "{exp_conf["exp_name"]}"...', flush=True)
print(f'Experiment Configuration: {exp_conf}\n', flush=True)

# Use GPU
device = torch.device('cuda:0')

# Clean-Up Models Folder
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

### Create and run the model
# Create the model
model = caranet().to(device)
# Get the first round of parameters from the server
parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

# If mode is 'CV' n - 1 clients are used for training, and the leftover client data for centralized evaluating
if exp_conf['mode']=='CV':
    min_clients = exp_conf['n_clients'] - 1
    evaluate_path = f'../../../data/inputs/kvasir_federated/{exp_conf["n_clients"]}_flclients/flclient_{args.evaluate_client}'
else:
    min_clients = exp_conf['n_clients']
    evaluate_path = exp_conf['evaluate_path']

# Default parameters for the most basic FL Strategy (FedAvg)
strategy_params_dict = {
    'fit_metrics_aggregation_fn': agg_metrics
    , 'evaluate_fn': get_evaluate_fn(model, evaluate_path)
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

train_start_time = time.time()
# Start the FL Server
logging.getLogger('flwr').setLevel(logging.INFO)
result = fl.server.start_server(server_address="localhost:5040", strategy=fl_strategy, config=fl.server.ServerConfig(num_rounds=exp_conf['n_rounds']))
train_end_time = time.time()
elapsed_time = train_end_time - train_start_time

# Gather CV results and output to disk
if exp_conf['mode']=='CV':
    rounds_test_results = {key: np.array([metric for _, metric in result.metrics_centralized[key]]) for key in result.metrics_centralized.keys()} # Centralized
    rounds_test_results = {key: value[1:] for key, value in rounds_test_results.items()}
    best_model_round = rounds_test_results['evaluate_mdice'].argmax() + 1

    dict_cv_results = {
        'exp_name': [exp_conf['exp_name']]
        , 'k-folds': [exp_conf['n_clients']]
        , 'fold': [exp_conf['evaluate_client']]
        , 'federated_strategy': [exp_conf['federated_strategy']]
        , 'tau': [exp_conf['tau'] if 'tau' in exp_conf else None]
        , 'eta': [exp_conf['eta'] if 'eta' in exp_conf else None]
        , 'eta_l': [exp_conf['eta_l'] if 'eta_l' in exp_conf else None]
        , 'prox_mu': [exp_conf['prox_mu'] if 'prox_mu' in exp_conf else None]
        , 'elapsed_time': elapsed_time
        , 'best_eval_value': [rounds_test_results['evaluate_mdice'].max()]
        , 'best_round': best_model_round
    }

    df_cv_results = pd.DataFrame.from_dict(dict_cv_results, orient='columns')
    filepath = exp_conf['logs_path']

    if(filepath!=None):
        filename = f'{filepath}/hyperparameter_tuning.csv'
        df_cv_results.to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))
elif exp_conf['mode']=='Test':
    print(f'\n===FL Experiment {exp_conf["exp_name"]} done, performing Federated Testing===', flush=True)
    ### Final Model Testing
    # Format results
    rounds_train_results = {key: np.array([metric for _, metric in result.metrics_distributed_fit[key]]) for key in result.metrics_distributed_fit.keys()} # Federated
    rounds_test_results = {key: np.array([metric for _, metric in result.metrics_centralized[key]]) for key in result.metrics_centralized.keys()} # Centralized
    rounds_test_results = {key: value[1:] for key, value in rounds_test_results.items()}

    ## Retrieve best model and load it for inference
    # Note: argmax() returns a 0-based index and there are server_rounds + 1 results from the FL run
    best_model_round = rounds_test_results['evaluate_mdice'].argmax() + 1

    # Load stored weights for best model
    filepath_model = f'{exp_conf["models_path"]}/round-{best_model_round}-weights.npz'

    parameters = []
    with np.load(filepath_model) as weights_best_model:
        for parameter in weights_best_model:
            parameters.append(weights_best_model[parameter])

    # Create the model
    model = caranet().to(device)

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
    inference_results = evaluate(model, exp_conf['test_path'])

    # Clean staging directory after retrieving the best model
    filepath_stage = exp_conf['models_path']
    for file in os.listdir(filepath_stage):
        os.remove(f'{filepath_stage}/{file}')

    ### Persist results to disk
    # Discard the first test result (From the random client initialization) as it is not a result of FedAvg
    rounds_results = {**rounds_train_results, **rounds_test_results}

    # # Persist Results to Disk if a filepath is provided
    filepath = exp_conf['logs_path']

    # # filepath = None
    res = compile_results(rounds_results, inference_results, exp_conf, best_model_round, elapsed_time, filepath)

    print(f'===\nFL Experiment {exp_conf["exp_name"]} concluded===\n', flush=True)
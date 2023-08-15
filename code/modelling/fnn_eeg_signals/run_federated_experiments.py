import concurrent.futures
import subprocess
import os
import signal
import time
import json
import sys
import itertools

### == HELPERS ==
def terminate_subprocesses(signum, frame):
    # Terminate all subprocesses
    for process in process_list:
        process.terminate()
    sys.exit(0)

def run_experiment(federated_strategy, param_set, client_config, log_file):
    print('============================================================================================', flush=True)
    print(f'Running "{federated_strategy}" with params: {param_set} and {client_config} FL Clients', flush=True)
    print('============================================================================================', flush=True)

    # Run the Federated Clients
    with concurrent.futures.ThreadPoolExecutor() as executor:
        print('=== Starting FL Server ===', flush='True')
        ## Start the Federated Server and wait 15 seconds before summoning the clients
        server = subprocess.Popen(
            ["python", "run_federated_server.py", str(federated_strategy), str(client_config), str(param_set)]
            , stdout=log_file, stderr=log_file
        )

        process_list.append(server)
        time.sleep(15)

        print('\n=== Starting FL Clients ===', flush='True')
        ## Summon the clients
        for client_id in range(1, client_config + 1):
            print(f'Client {client_id} of {client_config} summonned...', flush='True')
            client = subprocess.Popen(
                ["python", "run_federated_client.py", str(client_config), str(client_id)]
                , stdout=log_file, stderr=log_file
            )
            process_list.append(client)

        print('\n=== Training Model ===', flush='True')

        # Wait for all processes to end
        for process in process_list:
            process.wait()

### == Configuration ==
# Experiment Configuration
logs_path = "../../../data/logs/experiments/eegsignals_ffn"

# Strategies Evaluated
federated_strategies = ["FedAvg", "FedAdam", "FedAdagrad", "FedYogi", "FedProx"]

# Client Configurations
clients = [2, 4, 8, 16]
process_list = []

# Hyperparameter Grid (Stratified)
# hyperparameter_grid = {
#     'FedAdam': {
#         'tau': ['10e-1']
#         , 'eta': ['10e-0']
#         , 'eta_l': ['10e-0']
#     }
#     , 'FedAdagrad': {
#         'tau': ['10e-2']
#         , 'eta': ['10e-2']
#         , 'eta_l': ['10e-0']
#     }
#     , 'FedYogi': {
#         'tau': ['10e-5']
#         , 'eta': ['10e-3']
#         , 'eta_l': ['10e-2']
#     }
#     , 'FedProx': {
#         'prox_mu': ['10e-0']
#     }
#     , 'FedAvg': {}
# }

# Hyperparameter Grid (Patient-Aware)
hyperparameter_grid = {
    'FedAdam': {
        'tau': ['10e-2']
        , 'eta': ['10e-1']
        , 'eta_l': ['10e-1']
    }
    , 'FedAdagrad': {
        'tau': ['10e-0']
        , 'eta': ['10e-3']
        , 'eta_l': ['10e-3']
    }
    , 'FedYogi': {
        'tau': ['10e-0']
        , 'eta': ['10e-3']
        , 'eta_l': ['10e-2']
    }
    , 'FedProx': {
        'prox_mu': ['10e-3']
    }
    , 'FedAvg': {}
}

# Create Log File
logfile_path = f'{logs_path}/testing_logs.txt'
if os.path.exists(logfile_path):
    os.remove(logfile_path)
    
log_file = open(logfile_path, 'w')

original_stdout = sys.stdout
sys.stdout = log_file

# == EXECUTABLE ==
# Register the termination signal handler
signal.signal(signal.SIGINT, terminate_subprocesses)

# Generate parameter set combinations from the Hyperparameter grid
parameter_sets = {}

for algorithm, params in hyperparameter_grid.items():
    combinations = list(itertools.product(*params.values()))
    param_names = list(params.keys())
    
    param_dicts = [{param_names[i]: combination[i] for i in range(len(param_names))} for combination in combinations]
    
    parameter_sets[algorithm] = param_dicts

## Trigger Execution
# For each federated strategy, train for each set of parameters, for each configuration of FL clients
for federated_strategy in parameter_sets.keys():
    param_sets_strategy = parameter_sets[federated_strategy]
    for param_set in param_sets_strategy:
        for client_config in clients:
            run_experiment(federated_strategy, json.dumps(param_set), client_config, log_file)

sys.stdout = original_stdout
log_file.close()
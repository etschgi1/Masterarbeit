import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
import questionary
import importlib
from datetime import datetime
import sys, os
import numpy as np

sys.path.append('../src/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from MolGraphNetwork import MolGraphNetwork
from scf_guess_tools import Backend



def train_model(config, dataset, basis):
    MGNN = MolGraphNetwork(dataset=dataset,
                           basis=basis,
                           backend=Backend.PY,
                            batch_size=config["batch_size"],
                            hidden_dim=config["hidden_dim"],
                            message_passing_steps=config["message_passing_steps"],
                            edge_threshold_val=config["edge_threshold_val"],
                            message_net_layers=config["message_net_layers"],
                            message_net_dropout=config["message_net_dropout"],
                            data_aug_factor=config["data_aug_factor"],
                            target="density",
                            verbose_level=1)
    MGNN.load_data()
    MGNN.train_model(num_epochs=config["num_epochs"],
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                    grace_epochs=config["grace_epochs"],
                    lr_args={"mode":"min",
                             "factor":config["lr_factor"],
                             "patience": config["lr_patience"],
                             "threshold": config["lr_threshold"],
                             "cooldown": config["lr_cooldown"],
                             "min_lr" : config["lr_min"]},
                    report_fn=tune.report if tune.is_session_enabled() else None)
    

def runhypertune(config_file):
    dataset = None # TODO
    basis = None # TODO
    config_module = importlib.import_module(f"tune_config.{config_file.replace('.py', '')}")
    search_space = config_module.search_space
    tuner = tune.Tuner(
        tune.with_parameters(train_model, dataset=dataset, basis=basis),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=10,  # Number of hyperparameter samples to try
            metric="loss",  # Metric to optimize
            mode="min",  # Minimize the metric
            scheduler=ASHAScheduler(
                time_attr="training_iteration",
                metric="loss",
                mode="min",
                max_t=30,  # Maximum number of training iterations
                grace_period=5,  # Initial iterations to run before starting to evaluate
                reduction_factor=3, # prune factor
            )
        ),
        run_config=air.RunConfig( 
            name=f"MGNN_{config_file}",
            local_dir=LOG_DIR
        )
    )
    results = tuner.fit()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.save(os.path.join(TUNE_CONFIG_PATH, f"results_{config_file}_{timestamp}.hyperres"))
    print("Best hyperparameters found were: ", results.get_best_result().config)


def select_file():
    files = [f for f in os.listdir(TUNE_CONFIG_PATH) if os.path.isfile(os.path.join(TUNE_CONFIG_PATH, f))]
    
    if not files:
        print("No files found in", TUNE_CONFIG_PATH)
        return None

    selected = questionary.select(
        "Choose a config file:",
        choices=files
    ).ask()

    if selected:
        print(f"Selected file: {selected}")
        return os.path.join(TUNE_CONFIG_PATH, selected)
    return None

if __name__ == "__main__": 
    TUNE_CONFIG_PATH = os.path.dirname(os.path.abspath(__file__)) + "/tune_config"
    LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + "/tune_logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    config_file = select_file().split('/')[-1]  # Get the filename only
    runhypertune(config_file) # may later be extended to choose datasets with cmd line args
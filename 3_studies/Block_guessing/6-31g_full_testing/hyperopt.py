import torch
import torch.nn as nn
import torch.optim as optim
import ray 
from ray import tune
from ray.tune import RunConfig
from ray.air import session
from ray.air.config import RunConfig
from ray.tune.schedulers import ASHAScheduler
import questionary
import importlib, json
from scf_guess_datasets import Qm9
from pprint import pprint

from datetime import datetime
import sys, os
import numpy as np

sys.path.append('../../')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from mgnn.MolGraphNetwork import MolGraphNetwork
from scf_guess_tools import Backend

from utils import find_repo_root
PROJECT_ROOT = find_repo_root()
print(f"Project root found at: {PROJECT_ROOT}")
NUM_CPU = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
NUM_GPU = 1 if torch.cuda.is_available() else 0
print(f"Using {NUM_CPU} CPUs and {NUM_GPU} GPUs for Ray Tune")
CONC_TASKS = 4

def hyperopt_train(config, dataset, basis):
    import sys
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))
    ctx     = tune.get_context()
    storage = ctx.get_storage()
    persistent_dir = storage.trial_fs_path
    print("persistent dir:", persistent_dir)
    save_path = os.path.join(persistent_dir, "model.pth")
    # assert os.path.exists(persistent_dir)
    molgraphnet = MolGraphNetwork(dataset=dataset,
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
                            verbose_level=1,
                            no_progress_bar=True)
    molgraphnet.load_data()
    loss_on_full_matrix = config.get("loss_on_full_matrix", False)
    molgraphnet.train_model(num_epochs=config["num_epochs"],
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                    grace_epochs=config["grace_epochs"],
                    lr_args={"mode":"min",
                             "factor":config["lr_factor"],
                             "patience": config["lr_patience"],
                             "threshold": config["lr_threshold"],
                             "cooldown": config["lr_cooldown"],
                             "min_lr" : config["lr_min"]},
                    report_fn=tune.report,
                    loss_on_full_matrix=loss_on_full_matrix, 
                    model_save_path= save_path)
    molgraphnet.save_model(save_path)
    print("Model using")
    pprint(config)
    print("finished training!")

def runhypertune(config_file, slurm_start=False):
    if os.path.exists("/home/dmilacher/datasets/data1"):
        dataset = Qm9("/home/dmilacher/datasets/data1", size = 500, val=0.1, test=0.1)
    else: 
        dataset = Qm9("/home/etschgi1/REPOS/Masterarbeit/datasets/QM9", size = 500, val=0.1, test=0.1)
    basis = BASIS_PATH
    config_module = importlib.import_module(f"tune_config.{config_file.replace('.py', '')}")
    search_space = config_module.search_space
    disc_comb, disc_params = count_discrete_combinations(search_space)
    print(f"Total discrete combinations: {disc_comb}, Discrete parameters: {disc_params}")
    num_samples = disc_comb * 2 if disc_comb < 50 else disc_comb
    num_samples = min(num_samples, 100)
    print(f"Number of samples to try: {num_samples}")
    print(f"Concurrent trials: {CONC_TASKS}") 
    if not slurm_start and input("Start Ray Tune with the above configuration? (y/n): ").strip().lower() != 'y':
        print("Aborting Ray Tune run.")
        return
    tuner = tune.Tuner(tune.with_resources(
        tune.with_parameters(hyperopt_train, dataset=dataset, basis=basis),
        resources={"cpu": NUM_CPU // CONC_TASKS, "gpu": NUM_GPU / CONC_TASKS},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,  # Number of hyperparameter samples to try
            metric="loss",  # Metric to optimize
            mode="min",  # Minimize the metric
            max_concurrent_trials=CONC_TASKS,
            scheduler=ASHAScheduler(
                time_attr="training_iteration",
                max_t=50,  # Maximum number of training iterations
                grace_period= 5,  # Initial iterations to run before starting to evaluate
                reduction_factor=3, # prune factor
            )
        ),
        run_config=RunConfig( 
            name=f"MGNN_{config_file}",
            storage_path=LOG_DIR,
        )
    )
    print(f"Starting Ray Tune with config: {config_file}")
    print(f"Search space keys: {list(search_space.keys())}")
    print("------------------------------------------------")
    try:
        results = tuner.fit()
    except KeyboardInterrupt:
        print("Ray Tune run interrupted by user -> saving current results.")
    print("------------------------------------------------")

    print("Best hyperparameters found were: ", results.get_best_result().config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RES_DIR, f"results_{config_file}_{timestamp}.json")
    all_results = [{ "config": r.config, "metrics": r.metrics } for r in results]
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    best_result = results.get_best_result()
    with open(os.path.join(RES_DIR, f"best_{config_file}_{timestamp}.json"), 'w') as f:
        json.dump({
            "config": best_result.config,
            "metrics": best_result.metrics,
        }, f, indent=4)
    print(f"Results saved to {result_path} and best result to {os.path.join(RES_DIR, f'best_{config_file}_{timestamp}.json')}")
    print("Ray Tune run completed successfully.")

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

def count_discrete_combinations(search_space):
    total = 1
    discrete_params = {}
    for k, v in search_space.items():
        if isinstance(v, ray.tune.search.sample.Categorical):
            num = len(v.categories)
            total *= num
            discrete_params[k] = num
    return total, discrete_params

if __name__ == "__main__": 
    TUNE_CONFIG_PATH = os.path.dirname(os.path.abspath(__file__)) + "/tune_config"
    LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + "/tune_logs"
    RES_DIR = os.path.dirname(os.path.abspath(__file__)) + "/tune_results"
    BASIS_PATH = os.path.join(PROJECT_ROOT, "scripts/6-31g_2df_p_custom_nwchem.gbs")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    args = sys.argv[1:]  # Get command line arguments
    if args:
        config_file = args[0]
        if not config_file.endswith('.py'):
            print("Please provide a valid Python config file.")
            sys.exit(1)
        if not os.path.isfile(os.path.join(TUNE_CONFIG_PATH, config_file)):
            print(f"Config file {config_file} does not exist in {TUNE_CONFIG_PATH}.")
            sys.exit(1)
        slurmstart = True
        NUM_CPU = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        print(f"SLURM gave me {NUM_CPU} CPUs for Ray Tune")
    else:
        config_file = select_file().split('/')[-1]  # Get the filename only
        slurmstart = False
    print(f"Using config file: {config_file}")
    runhypertune(config_file, slurmstart) # may later be extended to choose datasets with cmd line args
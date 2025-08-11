import torch
import torch.nn as nn
import torch.optim as optim
import importlib, json, questionary
from scf_guess_datasets import Qm9Isomeres
import scf_guess_datasets
from pathlib import Path

from datetime import datetime
import sys, os
import numpy as np
from pprint import pprint

sys.path.append('../../')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from mgnn.MolGraphNetwork import MolGraphNetwork
from scf_guess_tools import Backend, load

from utils import find_repo_root
PROJECT_ROOT = find_repo_root()
print(f"Project root found at: {PROJECT_ROOT}")
# NUM_CPU = os.cpu_count()//16 or 1  # Fallback to 1 if os.cpu_count() returns None
# NUM_GPU = 1 if torch.cuda.is_available() else 0
# print(f"Using {NUM_CPU} CPUs and {NUM_GPU} GPUs for Eval")

LOG_ROOT = os.path.join(PROJECT_ROOT, "3_studies/Block_guessing/6-31g_testing/tune_logs")
BASIS_PATH = os.path.join(PROJECT_ROOT, "scripts/6-31g_2df_p_custom_nwchem.gbs")

def load_using_config(config, dataset, basis, model_path): 
    print("Loading model from ", model_path)
    import sys
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))
    molgraphnet = MolGraphNetwork(dataset=dataset,
                           basis=basis,
                           backend=Backend.PY,
                            batch_size=config["batch_size"],
                            hidden_dim=config["hidden_dim"],
                            message_passing_steps=config["message_passing_steps"],
                            edge_threshold_val=config["edge_threshold_val"],
                            message_net_layers=config["message_net_layers"],
                            message_net_dropout=config["message_net_dropout"],
                            data_aug_factor=1, #-> not important for inference!!!
                            target="density",
                            verbose_level=1,
                            no_progress_bar=True)
    molgraphnet.load_data()
    molgraphnet.load_model(model_path)
    print("Model using")
    pprint(config)
    print("loaded!")
    return molgraphnet

def train_using_config(config, dataset, basis, save_path):
    print(f"Training model and saving to {save_path}", flush=True)
    import sys
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))
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
    print("Start model training", flush=True)
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
                    model_save_path=save_path,
                    loss_on_full_matrix=loss_on_full_matrix)
    molgraphnet.save_model(save_path)
    print("Model using")
    pprint(config)
    print("finished training!")
    return molgraphnet

def select_folder():
    files = [f for f in os.listdir(LOG_ROOT)]
    
    if not files:
        print("No files found in", LOG_ROOT)
        return None

    selected = questionary.select(
        "Choose a log folder:",
        choices=files
    ).ask()

    if selected:
        print(f"Selected file: {selected}")
        return os.path.join(LOG_ROOT, selected)
    return None

from pyscf import gto, dft

def create_mf_from_mol(mol: gto.Mole, xc: str = "b3lyp") -> dft.RKS:
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.build()  # ensures XC grid is initialized
    return mf

def build_fock_from_density(mf: dft.RKS, density):
    vj, vk = mf.get_jk(dm=density)
    vxc = mf.get_veff(mf.mol, dm=density)
    hcore = mf.get_hcore()
    return hcore + vj + vxc - 0.5 * vk

def energy_elec(fock, density, coreH): 
    return np.trace((fock+coreH) @ density)

def energy_err(e_pred, e_conv): 
    return e_conv - e_pred, e_pred/e_conv -1

def diis_rmse(overlap, density, fock): 
    """Eq 2.3 - Milacher"""
    E = fock @ density @ overlap - overlap @ density @ fock
    diis_rmse_ = np.sqrt(np.linalg.norm(E, ord='fro')**2 / (density.shape[0]**2))
    return diis_rmse_

def update_stats(metrics, cum_stat_key="cummulative_stats"):
    """Update cumulative statistics in the metrics dictionary."""
    if metrics[cum_stat_key] is None:
        metrics[cum_stat_key] = {k: {} for k in metrics.keys() if k != cum_stat_key}
    
    for key, values in metrics.items():
        if key != cum_stat_key:
            metrics[cum_stat_key][key]["mean"] = np.mean(values)
            metrics[cum_stat_key][key]["std"] = np.std(values)

def eval_model(model, dataset, eval_result_path, skip_iterations= False):
    print("Evaluating model - gathering results...")
    test_graphs = model.get_graphs("test")
    density_preds = model.predict(test_graphs)
    pred_focks = []
    pred_overlaps = []
    coreHs = []
    print("Calculating Fock matrices, coreHs and overlaps...", flush=True)
    for i, (pred_density, key) in enumerate(zip(density_preds, dataset.test_keys)):
        cur_mol = dataset.molecule(key)
        mf = create_mf_from_mol(cur_mol, xc="b3lypg")
        pred_focks.append(dataset.solver(key).get_fock(dm=pred_density))
        pred_overlaps.append(mf.get_ovlp())
        coreHs.append(mf.get_hcore())
    print("Done...", flush=True)
    metrics = {"cummulative_stats": None, "energy_abs": [], "iterations": [], "energy_rel": [], "diis": [], "rmse": []}

    # start with energy_abs
    print("Calculating metrics...")
    for density, fock, coreH, overlap, key in zip(density_preds, pred_focks, coreHs, pred_overlaps, dataset.test_keys):
        print(f"Evaluating key: {key}")
        # energy
        e_conv = energy_elec(dataset.solution(key).fock, dataset.solution(key).density, dataset.solution(key).hcore)
        e_pred = energy_elec(fock, density, coreH)
        abs_err, rel_err = energy_err(e_pred, e_conv)
        metrics["energy_abs"].append(abs_err)
        metrics["energy_rel"].append(rel_err)
        print(f"Key: {key}, Energy Conv: {e_conv:.6f}, Energy Pred: {e_pred:.6f}, Abs Err: {abs_err:.6f}, Rel Err: {rel_err:.6f}")
        # diis
        metrics["diis"].append(diis_rmse(overlap, density, fock))
        print(f"Key: {key}, DIIS RMSE: {metrics['diis'][-1]:.6f}")
        # rmse
        rmse = np.sqrt(np.mean((density - dataset.solution(key).density)**2))
        metrics["rmse"].append(rmse)
        print(f"Key: {key}, RMSE: {rmse:.6f}")
        # iterations
        if not skip_iterations:
            solver = dataset.solver(key)
            _, _, _, _, status = scf_guess_datasets.solve(solver, density.astype(np.float64))
            metrics["iterations"].append(status.iterations)
            print(f"Key: {key}, Iterations: {status.iterations}", flush=True)
        # update cummulative stats
        update_stats(metrics)
        with open(eval_result_path, "w") as f:
            json.dump(metrics, f, indent=4)

        if len(metrics["iterations"]) > 5 and np.mean(metrics["iterations"]) > 13.5: 
            print("abort - mean iteration too high -> uninteresting!")
            break

    print("Done...")


def main(tune_log_folder, param_paths_override=None, skip_iterations=False, reevaluate=True): 
    # get all params
    all_params_path = [os.path.join(tune_log_folder, run, "params.json")  for run in os.listdir(tune_log_folder) if os.path.isdir(os.path.join(tune_log_folder, run))]
    if param_paths_override is not None:
        override_prefixes = [ov.split(",", 1)[0] for ov in param_paths_override]
        all_params_path = [p for p in all_params_path
                if any(Path(p).parent.name.startswith(pref) for pref in override_prefixes)
            ]
    #sort paths
    all_params_path.sort()
    # dataset = Qm9Isomeres("/home/dmilacher/datasets/data", size = 500, val=0.1, test=0.1)
    dataset = Qm9Isomeres("/home/etschgi1/REPOS/Masterarbeit/datasets/QM9", size = 500, val=0.1, test=0.1)

    basis = BASIS_PATH
    print(f"Dataset: {dataset.name}, for {len(all_params_path)} models", flush=True)

    for param_path in all_params_path: 
        # check if model already trained?
        try:
            with open(param_path, "r") as f:
                cur_config = json.load(f)
        except FileNotFoundError: 
            print(f"Parameter file {param_path} not found, skipping...")
            continue
        print(cur_config)
        model_path =  param_path.replace("params.json", "model.pth")
        eval_res_path = param_path.replace("params.json", "eval_res.json")
        if reevaluate and os.path.exists(eval_res_path):
            eval_res_path = eval_res_path.replace("eval_res.json", f"eval_res_reeval.json")
        if os.path.exists(eval_res_path) and not reevaluate:
            print(f"Evaluation results already exist at {eval_res_path}, skipping...")
            continue
        if os.path.exists(model_path):
            cur_model = load_using_config(cur_config, dataset, basis, model_path)
        else: 
            print("TRAIN MODEL")
            cur_model = train_using_config(cur_config, dataset, basis, model_path)
        eval_model(cur_model, dataset, eval_res_path, skip_iterations)

if __name__ == "__main__": 
    print("Starting evaluation script...", flush=True)
    param_paths_override = None
    skip_iterations = False
    if len(sys.argv) > 1:
        tune_log_folder = sys.argv[1]
    else:
        tune_log_folder = "/home/etschgi1/REPOS/Masterarbeit/3_studies/Block_guessing/6-31g_testing/tune_logs/MGNN_hyp_small_full_mat_loss.py"#select_folder()
        skip_iterations = True if input("skip iterations in benchmark?").lower() == "y" else False
    # check if param_paths_override.txt exists
    if os.path.exists(os.path.join(tune_log_folder, "param_paths_override.txt")):
        with open(os.path.join(tune_log_folder, "param_paths_override.txt"), "r") as f:
            param_paths_override = [line.strip() for line in f.readlines()]
        print(f"Using parameter paths override: {param_paths_override}")
    main(tune_log_folder, param_paths_override=param_paths_override, skip_iterations=skip_iterations)
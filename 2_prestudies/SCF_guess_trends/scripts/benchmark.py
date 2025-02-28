from scf_guess_tools import Metric, Psi4Engine, PySCFEngine
from util import *
import pandas as pd
import os
import numpy as np
import time
import multiprocessing
import re
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "8"
ENGINE = Psi4Engine(cache=True)
TIMEOUT_SECONDS = 60 # per molecule for calculation
# GUESS_METHODS = ['CORE', 'SAD', 'SADNO', 'GWH', 'HUCKEL', 'MODHUCKEL', 'SAP', 'SAPGAU'] 
# all Psi4 Methods
# GUESS_METHODS = ['minao', '1e', 'atom', 'huckel', 'vsap'] # all PySCF Methods

def getLetholaScores(basepath="2_prestudies/SCF_guess_trends/data/tables", metric_overwrite=None): 
    scores = pd.DataFrame()
    table_names = [os.path.join(basepath, filename) for filename in os.listdir(basepath) if filename.endswith(".txt")]
    for table_name in table_names: 
        try:
            df = pd.read_csv(table_name, sep="\s+", skiprows=2)
            df.columns = [col.lower() for col in df.columns]
            df = df[df["molecule"] != "Best"]
            with open(table_name) as f: 
                meta_line = f.readline().strip()
            id_ =  int(re.search("Table S[0-9]+", meta_line).group()[-1:])
            type_name = meta_line.split(":")[1].split(", ")[0]
            method = meta_line.split(", ")[-1].split("/")[0].strip()
            base = meta_line.split(f"{method}/")[-1].split(":")[0]
            benchmark_metric =  meta_line.split(f"{method}/")[-1].split(":")[-1].strip()

            df["table_id"] = id_
            df["type_name"] = type_name
            df["method"] = method
            df["base"] = base
            df["benchmark_metric"] = benchmark_metric if metric_overwrite is None else metric_overwrite

            scores = pd.concat([scores, df])
        except Exception as e:
            print(f"Fehler beim Lesen von {None}: {e}")
    return scores


def process_molecule(molecule_name, base, benchmark_metric, geometries_base_path, result_dict):
    """Handles the calculation and scoring for a single molecule with force-terminable process."""
    try:
        start = time.time()
        
        assert benchmark_metric in [metric.value for metric in Metric], f"Metric {benchmark_metric} not supported."
        
        molecule_path = next(
            (os.path.join(root, file) for root, _, files in os.walk(geometries_base_path) 
             for file in files if file == f"{molecule_name}.xyz"), None
        )
        if molecule_path is None:
            raise FileNotFoundError(f"XYZ file for molecule {molecule_name} not found.")
        
        molecule = ENGINE.load(molecule_path)
        final_res = ENGINE.calculate(molecule, basis=base)
        
        # Compute scores for all guessing methods
        scores = {}
        for guess_method in ENGINE.guessing_schemes:
            guess_res = ENGINE.guess(molecule, basis=base, method=guess_method)
            score = ENGINE.score(guess_res, final_res, metric=Metric(benchmark_metric))
            scores[guess_method] = score  # Store results
        
        end = time.time()
        scores["time"] = end - start  # Store computation time
        
        result_dict.update(scores)  # Save results in shared dict
        
    except Exception as e:
        print(f"Calculation failed for {molecule_name}: {e}")
        result_dict.clear()  # Indicate failure


def getOwnScores(molecules, checkpoint_file: Optional[str] = None, geometries_base_path="2_prestudies/SCF_guess_trends/data/geometries"):
    """Calculates the given method score for all molecules in the df with checkpointing and timeout support."""
    
    scores = molecules.copy()


    # Keep only relevant columns
    scores = scores[["table_id", "molecule", "type_name", "method", "base", "benchmark_metric"]]
    
    # Add empty columns for the scores
    for guess_method in ENGINE.guessing_schemes:
        if guess_method not in scores.columns:
            scores[guess_method] = pd.Series(dtype='float64')
    
    # Add time column
    if "time" not in scores.columns:
        scores["time"] = pd.Series(dtype='float64')
    
    # If checkpoint file exists, load progress
    start_index = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        scores = pd.read_csv(checkpoint_file)
        start_index = scores[scores["time"].notna()].index[-1] + 1  # Resume from last completed entry

    nr_molecules = len(molecules["molecule"])
    index = start_index + 1 if checkpoint_file else 1

    # Loop through molecules, skipping already processed ones
    for molecule_name, method, base, benchmark_metric in zip(
        molecules["molecule"][start_index:], molecules["method"][start_index:], 
        molecules["base"][start_index:], molecules["benchmark_metric"][start_index:]
    ):
        print(f"{index}/{nr_molecules}", molecule_name, method, base, benchmark_metric)
        index += 1
        
        result = None

        # Use multiprocessing.Manager to share result
        with multiprocessing.Manager() as manager:
            result_dict = manager.dict()
            process = multiprocessing.Process(
                target=process_molecule, 
                args=(molecule_name, base, benchmark_metric, geometries_base_path, result_dict)
            )
            process.start()
            process.join(timeout=TIMEOUT_SECONDS)

            if process.is_alive():
                print(f"Timeout reached for {molecule_name}. Killing process.")
                process.terminate()
                process.join()  # Ensure termination
                continue  # Skip molecule

            if result_dict:
                # Store results in DataFrame
                for key, value in result_dict.items():
                    scores.loc[
                        (scores["molecule"] == molecule_name) & 
                        (scores["method"] == method) & 
                        (scores["base"] == base), key
                    ] = value

        # Save checkpoint after each molecule
        if checkpoint_file:
            scores.to_csv(checkpoint_file, index=False)

    return scores


if __name__ == "__main__": 
    print(ENGINE.guessing_schemes)
    lethola_scores = getLetholaScores(metric_overwrite="f-score")
    lethola_scores.to_csv("2_prestudies/SCF_guess_trends/data/lethola_all.csv", index=False)
    molecule_paths = get_xyz_file_paths("2_prestudies/SCF_guess_trends/data/geometries")
    own_scores = getOwnScores(lethola_scores, checkpoint_file = "2_prestudies/SCF_guess_trends/data/own_all.csv")
    own_scores.to_csv("2_prestudies/SCF_guess_trends/data/own_all.csv", index=False)
    print("Done with all calculations.")
    # print(len(files))

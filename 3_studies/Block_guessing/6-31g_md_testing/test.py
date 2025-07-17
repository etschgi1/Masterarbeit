import os
import torch
import torch.nn as nn
import torch.optim as optim
import ray 
from ray import tune
from ray.tune import RunConfig
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import questionary
import importlib, json
from scf_guess_datasets import Qm9IsomeresMd
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

name = "hyperopt_train_36255_00000_0_batch_size=8,data_aug_factor=4,edge_threshold_val=3.5782,hidden_dim=512,lr=0.0004,message_net_dropout_2025-07-16_10-31-04/"
print(os.path.exists(os.path.join(PROJECT_ROOT, "3_studies/Block_guessing/6-31g_md_testing/tune_logs/MGNN_hyp_small_md.py", name)))
import sys, os, json
import numpy as np
sys.path.append('../src/')
from scf_guess_datasets import Qm9Isomeres
import matplotlib.pyplot as plt
from utils import find_repo_root
from scf_guess_tools import Backend


from pyscf import gto
from pyscf import scf
from pyscf import soscf

project_root_dir = find_repo_root()
print("Project root directory:", project_root_dir)

BASIS_PATH = os.path.join(project_root_dir, "scripts/6-31g_2df_p_custom_nwchem.gbs")
save_path = os.path.join(project_root_dir, "3_studies/second_order_conv/saves")

dataset = Qm9Isomeres(
    "/home/dmilacher/datasets/data", 
    size = 500, # number of molecules to load
    val=0.1, # using 80% training / 10 % val / 10% test split
    test=0.1,
)

densities = []
for key in dataset.train_keys: 
    sample = dataset.solution(key)
    densities.append(sample.density)
densities = np.array(densities)
mean_density = np.mean(densities, axis=0)


pyscfschemes = ["minao", "atom", "vsap", "1e"]


def so_sol(mol, guess, xc="b3lypg"):
    mf = scf.RKS(mol)
    mf.xc = xc
    mf.init_guess = guess
    mf.verbose = 4
    mf.max_cycle = 50
    mf = mf.newton() # second order solver
    kf_per_macro = []
    jk_per_macro = []
    def cb(loc):
        kf_per_macro.append(loc['kfcount'])
        jk_per_macro.append(loc['jkcount'])

    mf.callback = cb
    mf.kernel()
    return kf_per_macro, jk_per_macro

def so_sol_own(mol, guess, xc="b3lypg"):
    mf = scf.RKS(mol)
    mf.xc = xc
    mf.verbose = 4
    mf.max_cycle = 50
    mf = mf.newton() # second order solver
    kf_per_macro = []
    jk_per_macro = []
    def cb(loc):
        kf_per_macro.append(loc['kfcount'])
        jk_per_macro.append(loc['jkcount'])

    mf.callback = cb
    mf.kernel(dm0=guess)
    return kf_per_macro, jk_per_macro

res = {}

# for scheme in pyscfschemes:
#     res[scheme] = []
#     for key in dataset.test_keys:
#         print(f"Processing {key} with scheme {scheme}")
#         mol = dataset.molecule(key)
#         jk, kf = so_sol(mol, scheme)
#         res[scheme].append((jk, kf))
#         with open(os.path.join(save_path, f"second_order.json"), "w") as f:
#             json.dump(res, f, indent=4)
#     print("----")

from mgnn.MolGraphNetwork import MolGraphNetwork

MGNN = MolGraphNetwork(dataset=dataset,
                       basis=BASIS_PATH,
                       backend=Backend.PY,
                       batch_size=16,
                       hidden_dim=256,
                       message_passing_steps=4,
                       edge_threshold_val=3,
                       message_net_layers=3,
                       message_net_dropout=0.15,
                       target="density",
                       data_aug_factor=1,
                       verbose_level=2)
MGNN.load_data()
MGNN.load_model(f"{project_root_dir}/3_studies/Block_guessing/models/MGNN_6-31G_NO_AUG_07_07_manual_ref.pth")
test_graphs = MGNN.get_graphs("test")
density_preds = MGNN.predict(test_graphs, include_target=False, transform_to_density=True)
#GNN guess
res["GNN"] = []
for i, key in enumerate(dataset.test_keys):
    print(f"Processing {key} with GNN scheme")
    mol = dataset.molecule(key)
    jk, kf = so_sol_own(mol, density_preds[i])
    res["GNN"].append((jk, kf))
    with open(os.path.join(save_path, f"second_order_own.json"), "w") as f:
        json.dump(res, f, indent=4)
print("---- GNN scheme done ----")

# 0d scheme
res["0D"] = []
for key in dataset.test_keys:
    print(f"Processing {key} with 0d scheme")
    mol = dataset.molecule(key)
    jk, kf = so_sol_own(mol, mean_density)
    res["0D"].append((jk, kf))
    with open(os.path.join(save_path, f"second_order_own.json"), "w") as f:
        json.dump(res, f, indent=4)
print("---- 0D scheme done ----")
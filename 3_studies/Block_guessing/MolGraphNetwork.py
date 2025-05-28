# GNS example: https://medium.com/stanford-cs224w/graph-neural-network-based-simulator-predicting-particulate-and-fluid-systems-08ed0a20b28d

import os
from rdkit.Chem import rdmolfiles
from scf_guess_tools import Backend, load
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.data import Data, Batch

cur_path = os.path.dirname(__file__)
os.chdir(cur_path)
scripts_paths = ["../../scripts", "../"]
[sys.path.append(p) for p in scripts_paths if p not in sys.path]
from to_cache import density_fock_overlap
from BlockMatrix import BlockMatrix, Block


## Defines
ATOM_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

BASIS_PATH = "../../scripts/6-31g_2df_p_custom_nwchem.gbs"
GEOMETRY_Source = "../../datasets/QM9/xyz_c7h10o2_sorted"


class MolGraphNetwork(): 
    """A class to controll the GNN for density matrix prediction."""

    def __init__(self, xyz_source=GEOMETRY_Source, backend=Backend.PY, basis=BASIS_PATH):

        self.xyz_source = xyz_source
        self.xyz_files = [os.path.join(xyz_source, f) for f in os.listdir(xyz_source) if f.endswith('.xyz')]
        self.backend = backend
        self.basis = basis
        self.molgraphs = []


    def load_data(self, train_fraction=0.8, seed=42, max_samples=10, cache_meta={"method":"dft", "basis":BASIS_PATH, "functional": "b3lypg", "guess": "minao", "backend": "pyscf", "cache": "../../datasets/QM9/out/c7h10o2_b3lypg_6-31G(2df,p)/pyscf"}):
        """Load data from source directory split into train and test sets and create normalized BlockMatrices."""
        #! TODO Data augmentation
        # print(f"Loading {len(self.xyz_files)} files from {self.xyz_source}...")
        focks_in, dens_in, overlap_in, coords_in, atomic_nums_in = [], [], [], [], []
        for xyz_file in tqdm(self.xyz_files, desc="Loading files"):
            mol_name = os.path.basename(xyz_file).strip()
            # print(f"Using: {xyz_file}, {mol_name}, {cache_meta}")
            cached_ret = density_fock_overlap(filepath=xyz_file,
                                              filename = mol_name,
                                              method = cache_meta["method"],
                                              basis = None,
                                              functional = cache_meta["functional"],
                                              guess = cache_meta["guess"],
                                              backend = cache_meta["backend"],
                                              cache = cache_meta["cache"])
            
            if any([r == None for r in cached_ret]): 
                print(f"File {mol_name} bad - skipping")
                continue
            dens_in.append(cached_ret[0].numpy)
            focks_in.append(cached_ret[1].numpy)
            overlap_in.append(cached_ret[2].numpy)
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
                coords = [list(map(float, line.split()[1:4])) for line in lines[2:-3]]
                assert len(coords) == int(lines[0]), f"Number of coordinates {len(coords)} does not match number of atoms {int(lines[0])} in file {xyz_file}"
                coords_in.append(coords)
                atomic_nums = np.array([ATOM_NUMBERS[line.split()[0]] for line in lines[2:-3]])
                assert len(atomic_nums) == int(lines[0]), f"Number of atomic numbers {len(atomic_nums)} does not match number of atoms {int(lines[0])} in file {xyz_file}"
                atomic_nums_in.append(atomic_nums)

            if max_samples != None and len(coords) >= max_samples: 
                print(f"Reached max_samples ({max_samples}) for {xyz_file}. Early stopping.")
                break
        
        for overlap, coords, atomic_nums, xyz_file in tqdm(zip(overlap_in, coords_in, atomic_nums_in, self.xyz_files), desc="Creating graphs"):
            mol = load(xyz_file, backend=self.backend, basis=self.basis).native
            self.molgraphs.append(self.make_graph(overlap, coords, atomic_nums, mol, edge_threshold_type="max", edge_threshold=1e-8))
            
        #! TODO split into train and test sets & Norm!!!

    def make_graph(self, S, coords, atomic_nums, mol, max_block_dim = 26, edge_threshold_type="max", edge_threshold=1e-8): 
        """Create a graph from the overlap matrix S, coordinates, and atomic numbers. edge_threshold_type can be 'max' or 'mean' to determine the threshold for edges if func >= edge_threshold."""
        assert edge_threshold_type in ["max", "mean"], "edge_threshold_type must be 'max' or 'mean'."
        S_bm = BlockMatrix(mol, Matrix=S)
        blocks = S_bm.blocks

        # Let's start with the node features! 
        # node should include atomic number, overlap (center) block
        # maybe contain: orbitals one hot encoded?! -> maybe this is too redundant for our usecase
        # maybe contain: coordinates: absolute coordinates are not of importance but maybe their relation is?
        center_blocks = S_bm.get_blocks_by_type("center")
       

       # Build edges
       # build them according to the threshold criteria (max or mean overlap >= edge_threshold)
       # Include: Overlap block of the two atoms, distance between the atoms
       # Maybe include: some sort of angular / directional information?
       # Maybe include: difference in partial charges (we have this data in xyz?!)

if __name__ == "__main__": 

    MGNN = MolGraphNetwork(xyz_source=GEOMETRY_Source, backend=Backend.PY, basis=BASIS_PATH)
    MGNN.load_data()
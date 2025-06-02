# GNS example: https://medium.com/stanford-cs224w/graph-neural-network-based-simulator-predicting-particulate-and-fluid-systems-08ed0a20b28d

import os
from typing import List
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

    def __init__(self, 
                 xyz_source=GEOMETRY_Source,
                 max_block_dim=26,
                 hidden_dim=256,
                 backend=Backend.PY,
                 basis=BASIS_PATH, 
                 edge_threshold_type="atom_dist",
                 edge_threshold=5 # Angrom for "atom_dist", or dimensionless for "max" or "mean" 
                 ):

        self.xyz_source = xyz_source
        self.xyz_files = [os.path.join(xyz_source, f) for f in os.listdir(xyz_source) if f.endswith('.xyz')]
        self.backend = backend
        self.basis = basis
        self.molgraphs = []
        self.edge_threshold_type = edge_threshold_type  # Can be "max" or "mean" to determine the threshold for edges
        assert edge_threshold_type in ["atom_dist", "max", "mean", "fro"], "edge_threshold_type must be 'max' or 'mean'."
        self.edge_threshold = edge_threshold  # Default threshold for edge creation
        assert edge_threshold > 0, "edge_threshold must be a positive value."


        self.max_block_dim = 26  # Maximum number of atoms in a block (for QM9, this is 26)
        self.max_up = max_block_dim * (max_block_dim + 1) // 2  # Maximum size of upper triangular block
        self.max_sq = max_block_dim ** 2 

                # 1) NODE ENCODERS: one MLP per element
        self.node_encoders = torch.nn.ModuleDict({
            "C": torch.nn.Sequential(
                    torch.nn.Linear(self.max_up, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                 ),
            "O": torch.nn.Sequential(
                    torch.nn.Linear(self.max_up, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                 ),
            "H": torch.nn.Sequential(
                    torch.nn.Linear(self.max_up, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                 ),
        })

        # 2) EDGE ENCODERS: one MLP per unordered pair ("C_C", "C_O", "C_H", "O_O", "O_H", "H_H")
        self.edge_encoders = torch.nn.ModuleDict({
            "C_C": torch.nn.Sequential(
                      torch.nn.Linear(self.max_sq + 1, hidden_dim),
                      torch.nn.ReLU(),
                      torch.nn.Linear(hidden_dim, hidden_dim)
                   ),
            "C_O": torch.nn.Sequential(
                      torch.nn.Linear(self.max_sq + 1, hidden_dim),
                      torch.nn.ReLU(),
                      torch.nn.Linear(hidden_dim, hidden_dim)
                   ),
            "C_H": torch.nn.Sequential(
                      torch.nn.Linear(self.max_sq + 1, hidden_dim),
                      torch.nn.ReLU(),
                      torch.nn.Linear(hidden_dim, hidden_dim)
                   ),
            "O_O": torch.nn.Sequential(
                      torch.nn.Linear(self.max_sq + 1, hidden_dim),
                      torch.nn.ReLU(),
                      torch.nn.Linear(hidden_dim, hidden_dim)
                   ),
            "O_H": torch.nn.Sequential(
                      torch.nn.Linear(self.max_sq + 1, hidden_dim),
                      torch.nn.ReLU(),
                      torch.nn.Linear(hidden_dim, hidden_dim)
                   ),
            "H_H": torch.nn.Sequential(
                      torch.nn.Linear(self.max_sq + 1, hidden_dim),
                      torch.nn.ReLU(),
                      torch.nn.Linear(hidden_dim, hidden_dim)
                   ),
        })

        # 3) MESSAGE MLP (shared)
        self.message_net = torch.nn.Sequential(
            torch.nn.Linear(3 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # 4) NODE UPDATERS: one per element
        self.node_updaters = torch.nn.ModuleDict({
            "C": torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                 ),
            "O": torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                 ),
            "H": torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                 ),
        })

        # 5) DECODERS for reconstructing center‐blocks (upper‐triangle) and hetero‐blocks
        self.center_decoders = torch.nn.ModuleDict({
            "C": torch.nn.Linear(hidden_dim, self.max_up),
            "O": torch.nn.Linear(hidden_dim, self.max_up),
            "H": torch.nn.Linear(hidden_dim, self.max_up),
        })
        self.edge_decoders = torch.nn.ModuleDict({
            "C_C": torch.nn.Linear(hidden_dim, self.max_sq),
            "C_O": torch.nn.Linear(hidden_dim, self.max_sq),
            "C_H": torch.nn.Linear(hidden_dim, self.max_sq),
            "O_O": torch.nn.Linear(hidden_dim, self.max_sq),
            "O_H": torch.nn.Linear(hidden_dim, self.max_sq),
            "H_H": torch.nn.Linear(hidden_dim, self.max_sq),
        })


    def load_data(self, train_fraction=0.8, seed=42, max_samples=10, cache_meta={"method":"dft", "basis":BASIS_PATH, "functional": "b3lypg", "guess": "minao", "backend": "pyscf", "cache": "../../datasets/QM9/out/c7h10o2_b3lypg_6-31G(2df,p)/pyscf"}):
        """Load data from source directory split into train and test sets and create normalized BlockMatrices."""
        #! TODO Data augmentation
        # print(f"Loading {len(self.xyz_files)} files from {self.xyz_source}...")
        focks_in, dens_in, overlap_in, coords_in, atomic_nums_in = [], [], [], [], []
        
        if max_samples is not None and max_samples < len(self.xyz_files):
            print(f"Limiting to {max_samples} samples out of {len(self.xyz_files)} total files.")
            self.xyz_files = self.xyz_files[:max_samples]  # Limit to max_samples if specified
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
        
        for overlap, coords, atomic_nums, xyz_file in tqdm(zip(overlap_in, coords_in, atomic_nums_in, self.xyz_files), desc="Creating graphs"):
            mol = load(xyz_file, backend=self.backend, basis=self.basis).native
            self.molgraphs.append(self.make_graph(overlap, coords, atomic_nums, mol))
            
        #! TODO split into train and test sets & Norm!!!

    def make_graph(self, S, coords, atomic_nums, mol): 
        """Create a graph from the overlap matrix S, coordinates, and atomic numbers."""
        
        atom_slices = mol.aoslice_by_atom()
        n_atoms = len(atom_slices)

        # Let's start with the node features! 
        # node should include atomic number, overlap (center) block
        # maybe contain: orbitals one hot encoded?! -> maybe this is too redundant for our usecase
        # maybe contain: coordinates: absolute coordinates are not of importance but maybe their relation is?

        # overlap center blocks!
        center_blocks: List[torch.Tensor] = []
        for atom_index in range(n_atoms):
            _, _, ao_start, ao_end = atom_slices[atom_index]
            center = S[ao_start:ao_end, ao_start:ao_end]
            upper_tri = np.triu_indices(center.shape[0], k=0)  
            flat_center = center[upper_tri]  
            center_blocks.append(torch.from_numpy(flat_center).float())
        # Z is given by the atomic_nums lsit!
        center_sym = [f"{mol.atom_symbol(i)}_{mol.atom_symbol(i)}" for i in range(n_atoms)]  # e.g. "C_C", "O_O", "H_H"


       # Build edges
       # build them according to the threshold criteria (max or mean overlap >= edge_threshold)
       # Include: Overlap block of the two atoms, distance between the atoms
       # Maybe include: some sort of angular / directional information?
       # Maybe include: difference in partial charges (we have this data in xyz?!)

        edge_index_list = []
        edge_blocks: List[torch.Tensor] = []
        edge_dist: List[torch.Tensor] = []
        edge_pair_sym: List[str] = [] # e.g. "C_C", "C_O", "C_H", "O_O", "O_H", "H_H"

        S_tens = torch.from_numpy(S).float()

        def edge_threshold(block, coords=None): 
            if self.edge_threshold_type == "max":
                return block.max().item() >= self.edge_threshold
            elif self.edge_threshold_type == "mean":
                return block.mean().item() >= self.edge_threshold
            elif self.edge_threshold_type == "fro": 
                return torch.norm(block, p='fro').item() >= self.edge_threshold
            elif self.edge_threshold_type == "atom_dist":
                assert coords is not None, "coords must be provided when edge_threshold_type is 'atom_dist'."
                return float(np.linalg.norm(coords[0] - coords[1])) <= self.edge_threshold
            else:
                raise ValueError(f"Unknown edge_threshold_type: {self.edge_threshold_type}. Use 'max', 'mean', 'fro', or 'atom_dist'.")
        
        # build edges
        for i in range(n_atoms): 
            _, _, ai_start, ai_stop = atom_slices[i]
            n_i = ai_stop - ai_start
            for j in range(i + 1, n_atoms): # +1 to skip center blocks
                _, _, aj_start, aj_stop = atom_slices[j]
                n_j = aj_stop - aj_start

                block = S[ai_start:ai_stop, aj_start:aj_stop] # overlap block (homo / hetero depending on sym)
                coords_i, coords_j = np.array(coords[i]), np.array(coords[j])
                if not edge_threshold(block, coords=(coords_i, coords_j)): 
                    continue
                flat_ij = block.reshape(-1)
                edge_blocks.append(torch.from_numpy(flat_ij).float())
                # distance between atoms
                r_ij = float(np.linalg.norm(coords_i - coords_j))
                edge_dist.append(r_ij)

                key = "_".join(sorted([mol.atom_symbol(i), mol.atom_symbol(j)]))
                edge_pair_sym.append(key)
                
                # one direction
                edge_index_list.append([i,j])
                # as i understand we need two directed edges in pytorch geom - reuse same 
                edge_blocks.append(torch.from_numpy(flat_ij).float())
                edge_dist.append(r_ij)
                edge_pair_sym.append(key)
                edge_index_list.append([j, i])
        
        if len(edge_index_list) == 0: 
            raise ValueError("No edges found in the graph. Check your edge thresholding criteria.")
        else: 
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        # Finally we assemble our graph data object
        #! We build this manually because we have differntly sized center / overlap blocks for our different encoders and decoders! 
        data = Data(edge_index=edge_index)
        data.atom_sym      = center_sym           # List[str], length = n_atoms
        data.center_blocks = center_blocks       # List[Tensor], length = n_atoms
        data.edge_blocks   = edge_blocks         # List[Tensor], length = num_directed_edges
        data.edge_dist     = torch.tensor(edge_dist, dtype=torch.float)  # (num_directed_edges,)
        data.edge_pair_sym = edge_pair_sym       # List[str], length = num_directed_edges

if __name__ == "__main__": 

    MGNN = MolGraphNetwork(xyz_source=GEOMETRY_Source, backend=Backend.PY, basis=BASIS_PATH)
    MGNN.load_data(max_samples=1000)
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
from torch_geometric.loader import DataLoader

cur_path = os.path.dirname(__file__)
os.chdir(cur_path)
scripts_paths = ["../../../scripts", "../../"]
[sys.path.append(p) for p in scripts_paths if p not in sys.path]
from to_cache import density_fock_overlap
from BlockMatrix import BlockMatrix, Block
from utils import dprint

from encoder import EncoderDecoderFactory


## Defines
ATOM_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

BASIS_PATH = "../../../scripts/6-31g_2df_p_custom_nwchem.gbs"
GEOMETRY_Source = "../../../datasets/QM9/xyz_c7h10o2_sorted"


class MolGraphNetwork(torch.nn.Module): 
    """A class to controll the GNN for density matrix prediction."""

    def __init__(self, 
                 xyz_source=GEOMETRY_Source,
                 max_block_dim=26,
                 hidden_dim=256,
                 batch_size=32,
                 train_val_test_ratio = (0.8, 0.1, 0.1), # train, val, test
                 backend=Backend.PY,
                 basis=BASIS_PATH, 
                 edge_threshold_type="atom_dist",
                 edge_threshold_val=5, # Angrom for "atom_dist", or dimensionless for "max" or "mean" 
                 target="fock"
                 ):
        super().__init__()

        self.xyz_source = xyz_source
        self.xyz_files = [os.path.join(xyz_source, f) for f in os.listdir(xyz_source) if f.endswith('.xyz')]
        self.backend = backend
        self.basis = basis
        self.molgraphs = []
        self.edge_threshold_type = edge_threshold_type  # Can be "max" or "mean" to determine the threshold for edges
        assert edge_threshold_type in ["atom_dist", "max", "mean", "fro"], "edge_threshold_type must be 'max' or 'mean'."
        self.edge_threshold = edge_threshold_val  # Default threshold for edge creation
        assert edge_threshold_val > 0, "edge_threshold must be a positive value."
        assert target in ["fock", "density"], "target must be either 'fock' or 'density'."
        self.target = target

        self.train_ratio = train_val_test_ratio[0]
        self.val_ratio = train_val_test_ratio[1]
        self.test_ratio = train_val_test_ratio[2]
        # Instantiated in load_data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.max_block_dim = 26  # Maximum number of atoms in a block (for QM9, this is 26)
        self.max_up = max_block_dim * (max_block_dim + 1) // 2  # Maximum size of upper triangular block
        self.max_sq = max_block_dim ** 2 

        self.atom_types = None
        self.overlap_types = None

        # Encoder / Decoder factory to generate stuff for different atom types - instantiated in setup_model inside load_data
        self.node_encoders = None # encodes upper triangular blocks for center blocks
        self.node_updaters = None # updates node features based on messages
        self.center_decoders = None # decodes hidden node features to upper triangular blocks for centers
        self.edge_encoders = None # encodes hetero/homo blocks for edges
        self.edge_decoders = None # decodes hidden edge features to hetero/homo blocks for edges
        self.message_net = None # Net to provide message passing! 


    def load_data(self, seed=42, max_samples=10, cache_meta={"method":"dft", "basis":BASIS_PATH, "functional": "b3lypg", "guess": "minao", "backend": "pyscf", "cache": "../../../datasets/QM9/out/c7h10o2_b3lypg_6-31G(2df,p)/pyscf"}):
        """Load data from source directory split into train and test sets and create normalized BlockMatrices."""
        #! TODO Data augmentation
        # print(f"Loading {len(self.xyz_files)} files from {self.xyz_source}...")
        focks_in, dens_in, overlap_in, coords_in = [], [], [], []
        
        if max_samples is not None and max_samples < len(self.xyz_files):
            print(f"Limiting to {max_samples} samples out of {len(self.xyz_files)} total files.")
            self.xyz_files = self.xyz_files[:max_samples]  # Limit to max_samples if specified - don't care about bad ones meaning we will sample fewer if some are bad!
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
        
        if self.target == "fock":
            target_in = focks_in
        elif self.target == "density":
            target_in = dens_in
        # gather graphs! 
        for overlap, target, coords, xyz_file in tqdm(zip(overlap_in, target_in, coords_in, self.xyz_files), desc="Creating graphs"):
            mol = load(xyz_file, backend=self.backend, basis=self.basis).native
            self.molgraphs.append(self.make_graph(overlap, target, coords, mol))
            
        # Split into train and test sets
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(self.molgraphs))
        total_samples = len(self.molgraphs)
        train_size = int(total_samples * self.train_ratio)
        val_size = int(total_samples * self.val_ratio)
        test_size = total_samples - train_size - val_size
        
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]

        self.train_graphs = [self.molgraphs[i] for i in train_indices]
        self.val_graphs = [self.molgraphs[i] for i in val_indices]
        self.test_graphs = [self.molgraphs[i] for i in test_indices]
        print(f"Total samples: {total_samples}, Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Normalize 
        self.compute_normalization_factors()
        self.apply_normalization(self.train_graphs, distance_cutoff=True)
        self.apply_normalization(self.val_graphs, distance_cutoff=True)
        self.apply_normalization(self.test_graphs, distance_cutoff=True)
        print(f"Normalization factors computed and applied. Center stats: {self.center_norm}, Edge stats: {self.edge_norm}")

        # Build DataLoaders
        self.train_loader = DataLoader(self.train_graphs, batch_size=self.batch_size, shuffle=True) # we shuffle for training such that we see all graphs in a different order each epoch
        self.val_loader = DataLoader(self.val_graphs, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_graphs, batch_size=self.batch_size, shuffle=False)

        meta_info = self.setup_model()
        print(f"---\nModel setup (encoders / decoders message net) complete!")
        print(f"Total encoders / decoders / updaters: {meta_info['total']}, Node: {meta_info['node']} ({self.atom_types} - atom types) * 3 (enc, dec, update), Edge: {meta_info['edge']} ({self.overlap_types} - overlap types) * 2 (enc, dec).")

    def train(self, epochs=10, lr=1e-3, weight_decay=1e-5):
        #!TODO
        pass

    def forward(self): 
        pass

    def setup_model(self, encoder_type="default", decoder_type="default"):
        factory = EncoderDecoderFactory(
            atom_types=self.atom_types,
            hidden_dim=self.hidden_dim,
            max_up=self.max_up,
            max_sq=self.max_sq,
            message_layers=self.message_net_layers if hasattr(self, 'message_net_layers') else 2,  # Default to 2 layers if not set
            message_dropout=self.message_net_dropout if hasattr(self, 'message_net_dropout') else 0.0,  # Default to 0.0 if not set
        )
        # encoder / decoders
        self.node_encoders = factory.node_encoders
        self.node_updaters = factory.node_updaters
        self.center_decoders = factory.center_decoders
        self.edge_encoders = factory.edge_encoders
        self.edge_decoders = factory.edge_decoders

        # message_net
        self.message_net = factory.message_net
        
        encoder_dec_counts = {
            "node": len(self.node_encoders),
            "edge": len(self.edge_encoders),
            }
        encoder_dec_counts["total"] = 3 * encoder_dec_counts["node"] + 2 * encoder_dec_counts["edge"]
        return encoder_dec_counts

    def make_graph(self, S, T, coords, mol): 
        """Create a graph from the overlap matrix S, target matrix T (fock / density) coordinates, and atomic numbers."""
        
        atom_slices = mol.aoslice_by_atom()
        n_atoms = len(atom_slices)

        # Let's start with the node features! 
        # node should include atomic number, overlap (center) block
        # maybe contain: orbitals one hot encoded?! -> maybe this is too redundant for our usecase
        # maybe contain: coordinates: absolute coordinates are not of importance but maybe their relation is?

        # overlap & target center blocks!
        S_center_blocks: List[torch.Tensor] = []
        T_center_blocks: List[torch.Tensor] = []  
        for atom_index in range(n_atoms):
            _, _, ao_start, ao_end = atom_slices[atom_index]
            # overlap
            S_center = S[ao_start:ao_end, ao_start:ao_end]
            upper_tri = np.triu_indices(S_center.shape[0], k=0)  
            S_flat_center = S_center[upper_tri]  
            S_center_blocks.append(torch.from_numpy(S_flat_center).float())

            # target
            T_center = T[ao_start:ao_end, ao_start:ao_end]
            T_flat_center = T_center[upper_tri]  # Flatten the upper triangular part
            T_center_blocks.append(torch.from_numpy(T_flat_center).float())

        # Z is given by the atomic_nums lsit!
        atom_sym = [mol.atom_symbol(i) for i in range(n_atoms)]  # e.g. "C", "O", "H"


       # Build edges
       # build them according to the threshold criteria (max or mean overlap >= edge_threshold)
       # Include: Overlap block of the two atoms, distance between the atoms
       # Maybe include: some sort of angular / directional information?
       # Maybe include: difference in partial charges (we have this data in xyz?!)

        edge_index_list = []
        S_edge_blocks: List[torch.Tensor] = []
        T_edge_blocks: List[torch.Tensor] = []  # Edge blocks for target (fock / density)
        edge_dist: List[torch.Tensor] = []
        edge_pair_sym: List[str] = [] # e.g. "C_C", "C_O", "C_H", "O_O", "O_H", "H_H"

        def _pass_edge_threshold(block, coords=None): 
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

                # overlap edges
                S_block = S[ai_start:ai_stop, aj_start:aj_stop] # overlap S_block (homo / hetero depending on sym)
                coords_i, coords_j = np.array(coords[i]), np.array(coords[j])
                if not _pass_edge_threshold(S_block, coords=(coords_i, coords_j)): 
                    continue
                S_flat_ij = S_block.reshape(-1)
                S_edge_blocks.append(torch.from_numpy(S_flat_ij).float())

                # target edges
                T_block = T[ai_start:ai_stop, aj_start:aj_stop]
                T_flat_ij = T_block.reshape(-1)
                T_edge_blocks.append(torch.from_numpy(T_flat_ij).float())

                # distance between atoms
                r_ij = float(np.linalg.norm(coords_i - coords_j))
                edge_dist.append(r_ij)

                key = "_".join(sorted([mol.atom_symbol(i), mol.atom_symbol(j)]))
                edge_pair_sym.append(key)
                
                # one direction
                edge_index_list.append([i,j])
                # as i understand we need two directed edges in pytorch geom - reuse same 
                S_edge_blocks.append(torch.from_numpy(S_flat_ij).float()) #! no transpose here because source / target block change shouldn't matter for the model
                T_edge_blocks.append(torch.from_numpy(T_flat_ij).float())
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
        data.atom_sym = atom_sym
        data.center_blocks = S_center_blocks
        data.edge_blocks = S_edge_blocks
        data.edge_dist = torch.tensor(edge_dist, dtype=torch.float)
        data.edge_pair_sym = edge_pair_sym
        data.target_center_blocks = T_center_blocks
        data.target_edge_blocks = T_edge_blocks
        #! KEEP IN MIND: PyG won't move all attributes to GPU atuomatically!!!
        return data
    
    def apply_normalization(self, graph_list, distance_cutoff=True):
        """Apply normalization to the center and edge blocks of the graphs in graph_list.
        If distance_cutoff is True -> Edges will be in [0, self.edge_threshold] range."""
        for graph in graph_list:
            # source & target normalization!
            # centers
            for i, (S_center_block, T_center_block) in enumerate(zip(graph.center_blocks, graph.target_center_blocks)):
                key = graph.atom_sym[i]
                # source
                S_mean_t, S_std_t = torch.tensor(self.center_norm[key][0], device=S_center_block.device), torch.tensor(self.center_norm[key][1], device=S_center_block.device) # to use device block is stored on
                graph.center_blocks[i] = (S_center_block - S_mean_t) / S_std_t
                # target
                T_mean_t, T_std_t = torch.tensor(self.center_norm_target[key][0], device=T_center_block.device), torch.tensor(self.center_norm_target[key][1], device=T_center_block.device)
                graph.target_center_blocks[i] = (T_center_block - T_mean_t) / T_std_t
            # edges
            for i, (S_edge_block, T_edge_block) in enumerate(zip(graph.edge_blocks, graph.target_edge_blocks)):
                key = graph.edge_pair_sym[i]
                # source
                S_mean_t, S_std_t = torch.tensor(self.edge_norm[key][0], device=S_edge_block.device), torch.tensor(self.edge_norm[key][1], device=S_edge_block.device)
                graph.edge_blocks[i] = (S_edge_block - S_mean_t) / S_std_t
                # target
                T_mean_t, T_std_t = torch.tensor(self.edge_norm_target[key][0], device=T_edge_block.device), torch.tensor(self.edge_norm_target[key][1], device=T_edge_block.device)
                graph.target_edge_blocks[i] = (T_edge_block - T_mean_t) / T_std_t
            
            if distance_cutoff:
                graph.edge_dist = torch.clamp(graph.edge_dist, min=0, max=self.edge_threshold)  # Ensure distances are within [0, edge_threshold]
            graph.edge_dist = graph.edge_dist / self.edge_threshold  # Normalize distances to [0, 1]

    def apply_inverse_normalization(self, graph_list):
        """Apply inverse normalization to the center and edge blocks of the graphs in graph_list."""
        for graph in graph_list:
            # centers
            for i, (S_center_block, T_center_block) in enumerate(zip(graph.center_blocks, graph.target_center_blocks)):
                key = graph.atom_sym[i]
                S_mean_t, S_std_t = torch.tensor(self.center_norm[key][0], device=S_center_block.device), torch.tensor(self.center_norm[key][1], device=S_center_block.device)
                graph.center_blocks[i] = S_center_block * S_std_t + S_mean_t

                T_mean_t, T_std_t = torch.tensor(self.center_norm_target[key][0], device=T_center_block.device), torch.tensor(self.center_norm_target[key][1], device=T_center_block.device)
                graph.target_center_blocks[i] = T_center_block * T_std_t + T_mean_t
            # edges
            for i, (S_edge_block, T_edge_block) in enumerate(zip(graph.edge_blocks, graph.target_edge_blocks)):
                key = graph.edge_pair_sym[i]
                # source
                S_mean_t, S_std_t = torch.tensor(self.edge_norm[key][0], device=S_edge_block.device), torch.tensor(self.edge_norm[key][1], device=S_edge_block.device)
                graph.edge_blocks[i] = S_edge_block * S_std_t + S_mean_t
                # target
                T_mean_t, T_std_t = torch.tensor(self.edge_norm_target[key][0], device=T_edge_block.device), torch.tensor(self.edge_norm_target[key][1], device=T_edge_block.device)
                graph.target_edge_blocks[i] = T_edge_block * T_std_t + T_mean_t
            
            # not really needed for our predictions
            # graph.edge_dist = graph.edge_dist * self.edge_threshold
        return
    
    def compute_normalization_factors(self, zero_std_val = 1e-6):
        """Compute normalization factors for center and edge blocks & normalization for target blocks.
        zero_std_val is used to avoid division by zero in case of zero standard deviation (not likely to happen in our case but good to have)."""
        
        # gather atom sorts: (we loop over all atoms to also support non isomers!)
        center_keys = set()
        for graph in self.train_graphs:
            center_keys.update(graph.atom_sym)   
        # cerate possible edge keys
        edge_keys = set()
        for graph in self.train_graphs:
            edge_keys.update(graph.edge_pair_sym)
        center_keys = sorted(center_keys)
        edge_keys = sorted(edge_keys)
        self.atom_types = center_keys  # Store atom types for later use
        self.overlap_types = edge_keys  # Store overlap types for later use

        
        print(f"Found {len(center_keys)} center keys ({center_keys}) and {len(edge_keys)} edge keys ({edge_keys}) in the training set. -> Totaling {len(center_keys) + len(edge_keys)} unique encoder/decoder.")
        # source normalization factors
        self.center_norm = {key: (0,0) for key in center_keys}  
        self.edge_norm = {key: (0,0) for key in edge_keys}

        # target normalization factors
        self.center_norm_target = {key: (0,0) for key in center_keys}
        self.edge_norm_target = {key: (0,0) for key in edge_keys}  

        S_center_vals, S_edge_vals = {key: [] for key in center_keys}, {key: [] for key in edge_keys}
        T_center_vals, T_edge_vals = {key: [] for key in center_keys}, {key: [] for key in edge_keys}

        for graph in self.train_graphs: 
            # center blocks: 
            for center_block, center_key in zip(graph.center_blocks, graph.atom_sym): 
                S_center_vals[center_key] += center_block.tolist()
            for center_block, center_key in zip(graph.target_center_blocks, graph.atom_sym):
                T_center_vals[center_key] += center_block.tolist()
            # edge blocks:
            for edge_block, edge_key in zip(graph.edge_blocks, graph.edge_pair_sym):
                S_edge_vals[edge_key] += edge_block.tolist()
            for edge_block, edge_key in zip(graph.target_edge_blocks, graph.edge_pair_sym):
                T_edge_vals[edge_key] += edge_block.tolist()
                
        for key in self.center_norm.keys():
            assert len(S_center_vals[key]) > 0, f"No center blocks found for key {key}. Something must be off!"
            assert len(T_center_vals[key]) > 0, f"No target center blocks found for key {key}. Something must be off!"
            self.center_norm[key] = (np.mean(S_center_vals[key]), max(np.std(S_center_vals[key]), zero_std_val))
            self.center_norm_target[key] = (np.mean(T_center_vals[key]), max(np.std(T_center_vals[key]), zero_std_val))

        for key in self.edge_norm.keys():
            assert len(S_edge_vals[key]) > 0, f"No edge blocks found for key {key}. Check your edge thresholding criteria."
            assert len(T_edge_vals[key]) > 0, f"No target edge blocks found for key {key}. Check your edge thresholding criteria."
            self.edge_norm[key] = (np.mean(S_edge_vals[key]), max(np.std(S_edge_vals[key]), zero_std_val))
            self.edge_norm_target[key] = (np.mean(T_edge_vals[key]), max(np.std(T_edge_vals[key]), zero_std_val))
        return
        

if __name__ == "__main__": 

    MGNN = MolGraphNetwork(xyz_source=GEOMETRY_Source, backend=Backend.PY, basis=BASIS_PATH)
    MGNN.load_data(max_samples=10)
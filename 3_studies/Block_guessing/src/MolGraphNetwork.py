import os, copy
from typing import List
from itertools import chain
from rdkit.Chem import rdmolfiles
from scf_guess_tools import Backend, load
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from collections import defaultdict
from torch_scatter import scatter_add #! maybe use other aggregation functions later on

from utils import dprint, set_verbose, density_fock_overlap, unflatten_triang

from encoder import EncoderDecoderFactory


set_verbose(2)  # Set the verbosity level for debugging output

## Defines
ATOM_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


class MolGraphNetwork(torch.nn.Module): 
    """A class to controll the GNN for density matrix prediction."""

    def __init__(self, 
                 xyz_source,
                 basis, 
                 max_block_dim=26,
                 hidden_dim=256,
                 batch_size=32,
                 train_val_test_ratio = (0.8, 0.1, 0.1), # train, val, test
                 message_passing_steps=2, # Number of message passing steps
                 backend=Backend.PY,
                 edge_threshold_type="atom_dist",
                 edge_threshold_val=3, # Angrom for "atom_dist", or dimensionless for "max" or "mean" 
                 target="fock",
                 **kwargs
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

        assert isinstance(train_val_test_ratio, (list, tuple)) and len(train_val_test_ratio) == 3, "train_val_test_ratio must be a list or tuple of three values (train, val, test)."
        assert sum(train_val_test_ratio) == 1.0, "train_val_test_ratio must sum to 1.0."
        self.train_ratio = train_val_test_ratio[0]
        self.val_ratio = train_val_test_ratio[1]
        self.test_ratio = train_val_test_ratio[2]
        # Instantiated in load_data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.message_passing_steps = message_passing_steps  

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

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if hasattr(self, "verbose_level"): 
            print(f"Setting verbose level to: {self.verbose_level}")
            set_verbose(self.verbose_level)


    def load_data(self, seed=42, max_samples=10, cache_meta={"method":"dft", "basis":None, "functional": "b3lypg", "guess": "minao", "backend": "pyscf", "cache": "../../../datasets/QM9/out/c7h10o2_b3lypg_6-31G(2df,p)/pyscf"}):
        """Load data from source directory split into train and test sets and create normalized BlockMatrices."""
        self.molgraphs = [] #reset in case this is called again later!
        dprint(1, f"Loading {len(self.xyz_files)} files from {self.xyz_source}...")
        focks_in, dens_in, overlap_in, coords_in = [], [], [], []
        
        if max_samples is not None and max_samples < len(self.xyz_files):
            dprint(1, f"Limiting to {max_samples} samples out of {len(self.xyz_files)} total files.")
            self.xyz_files = self.xyz_files[:max_samples]  # Limit to max_samples if specified - don't care about bad ones meaning we will sample fewer if some are bad!
        assert os.path.exists(cache_meta["cache"]), f"Cache path {cache_meta['cache']} does not exist. Please create it first."
        for xyz_file in tqdm(self.xyz_files, desc="Loading files"):
            mol_name = os.path.basename(xyz_file).strip()
            # dprint(f"Using: {xyz_file}, {mol_name}, {cache_meta}")
            cached_ret = density_fock_overlap(filepath=xyz_file,
                                              filename = mol_name,
                                              method = cache_meta["method"],
                                              basis = None,
                                              functional = cache_meta["functional"],
                                              guess = cache_meta["guess"],
                                              backend = cache_meta["backend"],
                                              cache = cache_meta["cache"])
            
            if any([r == None for r in cached_ret]): 
                dprint(1, f"File {mol_name} bad - skipping")
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
        self.test_ground_truth = [target_in[i] for i in test_indices]  
        self.val_ground_truth = [target_in[i] for i in val_indices]
        self.train_ground_truth = [target_in[i] for i in train_indices]
        self.test_ovlp_mat = [overlap_in[i] for i in test_indices]
        self.val_ovlp_mat = [overlap_in[i] for i in val_indices]
        self.train_ovlp_mat = [overlap_in[i] for i in train_indices]
        self.files = {"train": [self.xyz_files[i] for i in train_indices],
                      "val": [self.xyz_files[i] for i in val_indices],
                      "test": [self.xyz_files[i] for i in test_indices]}
        dprint(1, f"Total samples: {total_samples}, Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Normalize 
        self.compute_normalization_factors()
        self.apply_normalization(self.train_graphs, distance_cutoff=True)
        self.apply_normalization(self.val_graphs, distance_cutoff=True)
        self.apply_normalization(self.test_graphs, distance_cutoff=True)
        dprint(1, "Normalization factors computed and applied.")
        dprint(2, f"Center stats: {self.center_norm}, Edge stats: {self.edge_norm}")

        # Build DataLoaders - this doesn't fly idk why
        # self.train_loader = DataLoader(self.train_graphs, batch_size=self.batch_size, shuffle=True, collate_fn = lambda b: b) # we shuffle for training such that we see all graphs in a different order each epoch
        # self.val_loader = DataLoader(self.val_graphs, batch_size=self.batch_size, shuffle=False, collate_fn = lambda b: b)
        # self.test_loader = DataLoader(self.test_graphs, batch_size=self.batch_size, shuffle=False, collate_fn = lambda b: b)

        # manual batching bc i can't get DataLoader to work with my custom Data objects (different sizes)
        self.train_loader = self.manual_batch(self.train_graphs, shuffle=True)
        self.val_loader = self.manual_batch(self.val_graphs, shuffle=False)
        self.test_loader = self.manual_batch(self.test_graphs, shuffle=False)
        # test batching
        first_train_batch = next(iter(self.train_loader))
        len_atom_sym = len(first_train_batch.atom_sym)
        len_edge_pair_sym = len(first_train_batch.edge_pair_sym)
        len_center_blocks = len(first_train_batch.center_blocks)
        print(f"First train batch: {len_atom_sym} atoms, {len_edge_pair_sym} edges, {len_center_blocks} center blocks.")

        meta_info = self.setup_model()
        dprint(1, f"---\nModel setup (encoders / decoders message net) complete!")
        dprint(2, f"Total encoders / decoders / updaters: {meta_info['total']}, Node: {meta_info['node']} ({self.atom_types} - atom types) * 3 (enc, dec, update), Edge: {meta_info['edge']} ({self.overlap_types} - overlap types) * 2 (enc, dec).")

    def manual_collate(self, batch):
        """Custom collate function to handle different sizes of center and edge blocks."""
        # offset indices!
        new_edge_index = []
        cum_offset = 0
        for g in batch:
            new_edge_index.append(g.edge_index + cum_offset)
            cum_offset += g.num_nodes

        # Jetzt alle verschobenen edge_index-Tensoren aneinanderhängen
        new_edge_index = torch.cat(new_edge_index, dim=1)

        batch_data = Data(edge_index=new_edge_index)
        batch_data.num_nodes = sum(g.num_nodes for g in batch)  
        batch_data.center_blocks = [cb for g in batch for cb in g.center_blocks]  
        batch_data.target_center_blocks = [tcb for g in batch for tcb in g.target_center_blocks]  
        batch_data.edge_blocks = [eb for g in batch for eb in g.edge_blocks]
        batch_data.target_edge_blocks = [teb for g in batch for teb in g.target_edge_blocks]
        batch_data.edge_dist = torch.cat([g.edge_dist for g in batch], dim=0)  
        batch_data.atom_sym = [sym for g in batch for sym in g.atom_sym]
        batch_data.edge_pair_sym = [eps for g in batch for eps in g.edge_pair_sym]
        batch_data.num_graphs = len(batch)  
        #! no offset here and not needed for training - only non-batched predictions supported currently - otherwise we would need to offset these slices as well!
        # batch_data.ao_slices = [g.ao_slices for g in batch]
        # batch_data.edge_ao_slices = [g.edge_ao_slices for g in batch]

        return batch_data

    def get_graphs(self, set_name="train"):
        """Return the normalized! test graphs."""
        assert set_name in ["train", "val", "test"], "set_name must be 'train', 'val', or 'test'."
        if set_name == "train":
            return self.train_graphs
        elif set_name == "val":
            return self.val_graphs
        elif set_name == "test":
            return self.test_graphs
    
    def get_files(self, set_name="train"):
        """Return the file paths for the specified set (train, val, test)."""
        assert set_name in ["train", "val", "test"], "set_name must be 'train', 'val', or 'test'."
        return self.files[set_name]
    

    def rebuild_matrix(self, pred_center_blocks, pred_edge_blocks, ao_slices, edge_ao_slices):
        N = sum(end-start for _, _, start, end in ao_slices)  
        out = np.zeros((N, N), dtype=np.float32)  
        # place center blocks
        for i, (_, _, start, end) in enumerate(ao_slices):
            flat_center_block = pred_center_blocks[i]
            out[start:end, start:end] = unflatten_triang(flat_center_block.numpy(), end - start)
        for i, ((start_i, end_i), (start_j, end_j)) in enumerate(edge_ao_slices):
            flat_edge_block = pred_edge_blocks[2*i] #! OC we have to index every second because we doubled the edges (directed edges - see make_graph) but not the indices! 
            block = flat_edge_block.reshape(end_i - start_i, end_j - start_j)  
            out[start_i:end_i, start_j:end_j] = block
            out[start_j:end_j, start_i:end_i] = block.T
        return out
        
    def predict(self, graphs: List[Data], inv_transform=True, raw=False, include_target=False, transform_to_density=False): 
        """
        Run inference on list of molecular graphs, returning either raw block embeddings
        or full reconstructed matrices, with optional de-normalization and Fock→density conversion.

        Args:
            graphs (List[Data]):
                A list of PyG Data objects. Each must have:
                - `center_blocks` / `edge_blocks` (input overlap blocks),
                - `target_center_blocks` / `target_edge_blocks` (target blocks),
                - `ao_slices` / `edge_ao_slices` for reconstruction.
            inv_transform (bool, optional):
                If True, apply the inverse of the training normalization to all source, target,
                and predicted blocks before returning. Default: True.
            raw (bool, optional):
                If True, return raw block lists:
                - without reconstruction to full matrices.
                If False, return reconstructed square matrices. Default: False.
            include_target (bool, optional):
                If True, include target blocks or target matrices alongside predictions in the output.
                Default: False.
            transform_to_density (bool, optional):
                If True *and* raw is False *and* the model's original target was a Fock matrix,
                transform the predicted (and optionally target) Fock matrices into
                density matrices.
                Default: False.

        Returns:
            List:
            - **raw=False, include_target=False**  
                `[ np.ndarray(matrix)_graph1, np.ndarray(matrix)_graph2, … ]`
            - **raw=False, include_target=True**  
                `[ (pred_matrix, tgt_matrix)_graph1, … ]`
            - **raw=True, include_target=False**  
                `[ (pred_center_blocks, pred_edge_blocks)_graph1, … ]`
            - **raw=True, include_target=True**  
                `[ (pred_center_blocks, pred_edge_blocks,
                    target_center_blocks, target_edge_blocks)_graph1, … ]`

        """
        assert self.node_encoders is not None, "Model not set up. Call setup_model() first."
        assert self.edge_encoders is not None, "Model not set up. Call setup_model() first."
        
        pred_matrices = []
        for graph in graphs:
            graph = graph.to(next(self.parameters()).device)  
            with torch.no_grad():
                graph = self.forward(graph)
                if inv_transform: 
                    graph = self.apply_inverse_normalization([graph])[0]
                pred_center_blocks = graph.pred_center_blocks
                pred_edge_blocks = graph.pred_edge_blocks
                if not raw:
                    ao_slices, edge_ao_slices = graph.ao_slices, graph.edge_ao_slices
                    if transform_to_density: 
                        ovlp = self.rebuild_matrix(graph.center_blocks, graph.edge_blocks, ao_slices, edge_ao_slices)
                        print(f"Ovlp: {ovlp[:10, :10]}")
                        nocc = sum([ATOM_NUMBERS[sym] for sym in graph.atom_sym])  #! this works only for closed-shell systems!
                    pred_rebuild = self.rebuild_matrix(pred_center_blocks, pred_edge_blocks, ao_slices, edge_ao_slices)
                    pred_rebuild = pred_rebuild if not transform_to_density else self.transform_to_density(pred_rebuild, ovlp, nocc)
                    if include_target:
                        target_center_blocks = graph.target_center_blocks
                        target_edge_blocks = graph.target_edge_blocks
                        target_rebuild = self.rebuild_matrix(target_center_blocks, target_edge_blocks, ao_slices, edge_ao_slices)
                        target_rebuild = target_rebuild if not transform_to_density else self.transform_to_density(target_rebuild, ovlp, nocc)
                        pred_matrices.append((pred_rebuild, target_rebuild))
                    else:
                        pred_matrices.append(pred_rebuild)
                else: #raw
                    if transform_to_density:
                        Warning("transform_to_density is True but raw is also True. This will not transform the target blocks to density matrices!")
                    if include_target:
                        target_center_blocks = graph.target_center_blocks
                        target_edge_blocks = graph.target_edge_blocks
                        pred_matrices.append((pred_center_blocks, pred_edge_blocks, target_center_blocks, target_edge_blocks))
                    else:
                        pred_matrices.append((pred_center_blocks, pred_edge_blocks))
        return pred_matrices
    
    def check_positive_definite(self, S, tol=1e-10):
        eigvals = np.linalg.eigvalsh(S)
        is_pd = np.all(eigvals > tol)
        return is_pd

    def density_from_fock(self, fock, overlap, nocc):
        from scipy.linalg import eigh
        assert self.check_positive_definite(overlap)
        _, C = eigh(fock, overlap)
        C_occ = C[:, :nocc]
        density = 2 * C_occ @ C_occ.T 
        return density

    def transform_to_density(self, fock_mat, ovlp_mat, nocc):
        """Transform Fock matrices to density matrices."""
        if self.target == "density":
            Warning("Model was trained on density matrices, so there is no need to set transform_to_density=True. Returning non-transformed matrices.")
            return fock_mat
        # transform Fock to density
        assert self.target == "fock", "Model must be trained on Fock matrices to use this method."
        assert isinstance(fock_mat, np.ndarray), "fock_matrices must be a numpy array - can only transform reconstructed matrices!"
        return self.density_from_fock(fock=fock_mat, overlap=ovlp_mat, nocc=nocc)
    
    def get_source_mat(self, set_name="test"): 
        """Get the source overlap matrices for the specified set (train, val, test)."""
        assert set_name in ["train", "val", "test"], "set_name must be 'train', 'val', or 'test'."
        if set_name == "train":
            return self.train_ovlp_mat
        elif set_name == "val":
            return self.val_ovlp_mat
        elif set_name == "test":
            return self.test_ovlp_mat

    def get_ground_truth(self, set_name="test"):
        """Get the ground truth matrices (fock / density - depending on self.target) for the specified set (train, val, test)."""
        assert set_name in ["train", "val", "test"], "set_name must be 'train', 'val', or 'test'."
        if set_name == "train":
            return self.train_ground_truth
        elif set_name == "val":
            return self.val_ground_truth
        elif set_name == "test":
            return self.test_ground_truth


    def manual_batch(self, graphs, shuffle=True):
        """Manually batch the graphs into a List."""
        batched_ = []
        if shuffle:
            np.random.shuffle(graphs)  # Shuffle the graphs if required
        for i in range(0, len(graphs), self.batch_size):
            batch_graphs = graphs[i:i + self.batch_size]
            batch_graphs = self.manual_collate(batch_graphs)
            batched_.append(batch_graphs)
        return batched_

    def forward(self, batch): 
        device = batch.edge_index.device  
        N_total = len(batch.atom_sym)  # Total number of atoms in the batch - i.e. center blocks
        E_total = batch.edge_index.size(1)  # Total number of edges in the batch


        # I) Encode node features (center blocks)
        atom_indices_dict = defaultdict(list)  
        unique_atom_syms = set(batch.atom_sym)  #! we actually stack same type of atoms for faster processing
        c = torch.zeros((N_total, self.hidden_dim), device=device) 
        for u_sym in unique_atom_syms: 
            atom_sym_indices = [i for i, sym in enumerate(batch.atom_sym) if sym == u_sym]
            atom_indices_dict[u_sym].extend(atom_sym_indices)
            raw_center_blocks = torch.stack(
                [batch.center_blocks[i].to(device) for i in atom_sym_indices], dim=0
            ) # This now has shape (Nr_of_atoms_for_this_sym, self.center_sizes[u_sym])
            assert raw_center_blocks.shape[1] == self.center_sizes[u_sym], f"Center block size {raw_center_blocks.shape[1]} does not match expected size {self.center_sizes[u_sym]} for atom type {u_sym}."
            c_sym = self.node_encoders[u_sym](raw_center_blocks) 
            c[atom_sym_indices] = c_sym  # in h we have the same dimensiom self.hidden_dim for all atoms in the batch

        # II) Encode edge features (edge blocks)
        unique_edge_keys = set(batch.edge_pair_sym)  # Unique edge types
        edge_indices_dict = defaultdict(list)  
        e = torch.zeros((E_total, self.hidden_dim), device=device)  # Edge features
        for key in unique_edge_keys:
            edge_key_indices = [i for i, sym in enumerate(batch.edge_pair_sym) if sym == key]
            edge_indices_dict[key].extend(edge_key_indices)  # Store indices for this edge type
            raw_edge_blocks = torch.stack(
                [batch.edge_blocks[i].to(device) for i in edge_key_indices], dim=0
            )
            distances = batch.edge_dist[edge_key_indices].to(device).view(-1, 1)  # Reshape distances to match edge blocks -> (Nr of edges for this key, 1)
            edge_inputs = torch.cat((raw_edge_blocks, distances), dim=1)  
            
            e_key = self.edge_encoders[key](edge_inputs)  
            e[edge_key_indices] = e_key  

        # III) Message passing
        src_nodes = batch.edge_index[0]  # Remember that we saved two edges for each pair (i,j) and (j,i) in edge_index!
        tgt_nodes = batch.edge_index[1]  
        for _round in range(self.message_passing_steps): 
            c_src = c[src_nodes]  
            c_tgt = c[tgt_nodes]

            msg_inp = torch.cat([c_src, c_tgt, e], dim=1) # input to message net: [c_u || c_v || e_{u→v}]
            m = self.message_net(msg_inp)  

            agg = torch.zeros((N_total, self.hidden_dim), device=device)  
            agg = scatter_add(m, tgt_nodes, dim=0, dim_size=N_total)  

            c_new = torch.zeros_like(c)
            for i, sym in enumerate(batch.atom_sym):
                old_and_agg = torch.cat([c[i], agg[i]], dim=0)  # 2*self.hidden_dim; This goes into our node updater!
                c_new[i] = self.node_updaters[sym](old_and_agg)  # Update node features with the aggregated messages
            c = c_new 
        
        # IV) Decode node features to center blocks
        pred_center_blocks = [None] * len(batch.center_blocks)  # Note that we do not use numpy arrays here because differnt blocks have different sizes!
        for sym in unique_atom_syms:
            atom_sym_indices = atom_indices_dict[sym] #reuse the indices from encoding
            c_sym_stack = torch.stack([c[i] for i in atom_sym_indices], dim=0)  # (Nr_of_atoms_for_this_sym, self.hidden_dim)
            center_decoded = self.center_decoders[sym](c_sym_stack)  # Decode to center blocks
            for i, idx in enumerate(atom_sym_indices):
                pred_center_blocks[idx] = center_decoded[i]
        
        # V) Decode edge features to edge blocks
        pred_edge_blocks = [None] * len(batch.edge_blocks) 
        for key in unique_edge_keys: 
            edge_key_indices = edge_indices_dict[key]
            e_key_stack = torch.stack([e[i] for i in edge_key_indices], dim=0)  # (Nr_of_edges_for_this_key, self.hidden_dim)
            edge_decoded = self.edge_decoders[key](e_key_stack)  
            for i, idx in enumerate(edge_key_indices):
                pred_edge_blocks[idx] = edge_decoded[i]
        
        # Attach to batch object: 
        batch.pred_center_blocks = pred_center_blocks
        batch.pred_edge_blocks = pred_edge_blocks

        dprint(3, "Forward pass complete!")
        return batch


    def setup_model(self, model_type="default"):
        if model_type == "default": 
            self.gather_block_size_stats()
            factory = EncoderDecoderFactory(
                atom_types = self.atom_types,
                edge_types = self.overlap_types,
                hidden_dim = self.hidden_dim,
                center_sizes = self.center_sizes,
                edge_sizes = self.edge_sizes,
                message_layers = self.message_net_layers if hasattr(self, 'message_net_layers') else 2,  # Default to 2 layers if not set
                message_dropout = self.message_net_dropout if hasattr(self, 'message_net_dropout') else 0.0,  # Default to 0.0 if not set
            )
        # encoder / decoders
        self.node_encoders = factory.node_encoders
        self.node_updaters = factory.node_updaters
        self.center_decoders = factory.center_decoders
        self.edge_encoders = factory.edge_encoders
        self.edge_decoders = factory.edge_decoders

        # message_net
        self.message_net = factory.message_net
        dprint(2, f"Message net: {self.message_net}")
        encoder_dec_counts = {
            "node": len(self.node_encoders),
            "edge": len(self.edge_encoders),
            }
        encoder_dec_counts["total"] = 3 * encoder_dec_counts["node"] + 2 * encoder_dec_counts["edge"]
        return encoder_dec_counts

    def gather_block_size_stats(self):
        """Gather statistics about the sizes of center and edge blocks in the training graphs."""
        assert len(self.train_graphs) > 0, "No training graphs found. Please load data first."
        self.center_sizes = {}
        self.edge_sizes = {}
        for graph in self.train_graphs:
            for i, atom_sym in enumerate(graph.atom_sym):
                if atom_sym not in self.center_sizes:
                    self.center_sizes[atom_sym] = graph.center_blocks[i].shape[0]
                    dprint(2, f"Found center block size {self.center_sizes[atom_sym]} for atom type {atom_sym}.")
            for i, edge_sym in enumerate(graph.edge_pair_sym):
                if edge_sym not in self.edge_sizes:
                    self.edge_sizes[edge_sym] = graph.edge_blocks[i].shape[0]
                    dprint(2, f"Found edge block size {self.edge_sizes[edge_sym]} for edge type {edge_sym}.")

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
        edge_ao_slices = []
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
                edge_ao_slices.append(((ai_start, ai_stop), (aj_start, aj_stop)))
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
        data.num_nodes = n_atoms 
        data.atom_sym = atom_sym
        data.center_blocks = S_center_blocks
        data.edge_blocks = S_edge_blocks
        data.edge_dist = torch.tensor(edge_dist, dtype=torch.float)
        data.edge_pair_sym = edge_pair_sym
        data.target_center_blocks = T_center_blocks
        data.target_edge_blocks = T_edge_blocks
        # for reconstruction we need the ao_slice information
        data.ao_slices = atom_slices
        data.edge_ao_slices = edge_ao_slices
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
        """Apply inverse normalization to the center and edge blocks of the graphs in graph_list and return a new list of graphs (leaves initial graphs unchanged).
        Unnormalize source / target and predictions"""
        inv_graphs = []
        for graph in graph_list:
            # centers
            inv_g = copy.copy(graph)
            inv_g.center_blocks = [cb.clone() for cb in graph.center_blocks]  
            inv_g.target_center_blocks = [tcb.clone() for tcb in graph.target_center_blocks]
            inv_g.edge_blocks = [eb.clone() for eb in graph.edge_blocks]
            inv_g.target_edge_blocks = [teb.clone() for teb in graph.target_edge_blocks]
            inv_g.pred_center_blocks = [cb.clone() for cb in graph.pred_center_blocks]
            inv_g.pred_edge_blocks = [eb.clone() for eb in graph.pred_edge_blocks]
            # rest shouldn't change + distance is of no importance for our predictions and benchmarking
            for i, (S_center_block, T_center_block, P_center_block) in enumerate(zip(inv_g.center_blocks, inv_g.target_center_blocks, inv_g.pred_center_blocks)):
                key = inv_g.atom_sym[i]
                S_mean_t, S_std_t = torch.tensor(self.center_norm[key][0], device=S_center_block.device), torch.tensor(self.center_norm[key][1], device=S_center_block.device)
                inv_g.center_blocks[i] = S_center_block * S_std_t + S_mean_t

                T_mean_t, T_std_t = torch.tensor(self.center_norm_target[key][0], device=T_center_block.device), torch.tensor(self.center_norm_target[key][1], device=T_center_block.device)
                inv_g.target_center_blocks[i] = T_center_block * T_std_t + T_mean_t

                inv_g.pred_center_blocks[i] = P_center_block * T_std_t + T_mean_t  # Uses same stats as target (from training set)
            # edges
            for i, (S_edge_block, T_edge_block, P_edge_block) in enumerate(zip(inv_g.edge_blocks, inv_g.target_edge_blocks, inv_g.pred_edge_blocks)):
                key = inv_g.edge_pair_sym[i]
                # source
                S_mean_t, S_std_t = torch.tensor(self.edge_norm[key][0], device=S_edge_block.device), torch.tensor(self.edge_norm[key][1], device=S_edge_block.device)
                inv_g.edge_blocks[i] = S_edge_block * S_std_t + S_mean_t
                # target
                T_mean_t, T_std_t = torch.tensor(self.edge_norm_target[key][0], device=T_edge_block.device), torch.tensor(self.edge_norm_target[key][1], device=T_edge_block.device)
                inv_g.target_edge_blocks[i] = T_edge_block * T_std_t + T_mean_t

                inv_g.pred_edge_blocks[i] = P_edge_block * T_std_t + T_mean_t  # Uses same stats as target (from training set)

            inv_graphs.append(inv_g)
            # not really needed for our predictions
            # graph.edge_dist = graph.edge_dist * self.edge_threshold
        return inv_graphs
    
    def setup_atom_edge_keys(self):
        # gather atom sorts: (we loop over all atoms to also support non isomers!)
        center_keys = set()
        for graph in self.train_graphs:
            center_keys.update(graph.atom_sym)   
        # cerate possible edge keys
        edge_keys = set()
        for graph in self.train_graphs:
            edge_keys.update(graph.edge_pair_sym)
        if hasattr(self, "use_all_data_for_atom_edge_keys") and self.use_all_data_for_atom_edge_keys:
            for graph in self.val_graphs:
                center_keys.update(graph.atom_sym)
                edge_keys.update(graph.edge_pair_sym)
            for graph in self.test_graphs:
                center_keys.update(graph.atom_sym)
                edge_keys.update(graph.edge_pair_sym)
        center_keys = sorted(center_keys)
        edge_keys = sorted(edge_keys)
        self.atom_types = center_keys  # Store atom types for later use
        self.overlap_types = edge_keys  # Store overlap types for later use
        dprint(1, f"Found {len(center_keys)} center keys ({center_keys}) and {len(edge_keys)} edge keys ({edge_keys}) in the training set. -> Totaling {len(center_keys) + len(edge_keys)} unique encoder/decoder.")
    
    def compute_normalization_factors(self, zero_std_val = 1e-6):
        """Compute normalization factors for center and edge blocks & normalization for target blocks.
        zero_std_val is used to avoid division by zero in case of zero standard deviation (not likely to happen in our case but good to have)."""
        
        self.setup_atom_edge_keys()

        # source normalization factors
        self.center_norm = {key: (0,0) for key in self.atom_types}  
        self.edge_norm = {key: (0,0) for key in self.overlap_types}

        # target normalization factors
        self.center_norm_target = {key: (0,0) for key in self.atom_types}
        self.edge_norm_target = {key: (0,0) for key in self.overlap_types}  

        S_center_vals, S_edge_vals = {key: [] for key in self.atom_types}, {key: [] for key in self.overlap_types}
        T_center_vals, T_edge_vals = {key: [] for key in self.atom_types}, {key: [] for key in self.overlap_types}

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
    
    def train_model(self, num_epochs=5, lr=1e-3, weight_decay=1e-5, device=None, model_save_path=None, grace_epochs=10):
        import torch.nn.functional as F
        from tqdm import tqdm
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=5,
                                                               min_lr=1e-6)
        history = {
            "train_loss": [],
            "val_loss":[],
            "test_loss":float,
            "lr":[],
        }
        best_val, no_imp_epochs = float('inf'), 0
        try: 
            for epoch in range(1, num_epochs + 1):
                self.train()
                total_train_loss = 0.0

                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]"):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    batch = self.forward(batch)

                    loss_center, loss_edge = 0.0, 0.0
                    for i in range(batch.num_nodes):
                        loss_center += F.mse_loss(batch.pred_center_blocks[i], batch.target_center_blocks[i].to(device))
                    for k in range(batch.num_edges):
                        loss_edge += F.mse_loss(batch.pred_edge_blocks[k], batch.target_edge_blocks[k].to(device))

                    loss = (loss_center + loss_edge) / batch.num_graphs
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                avg_train_loss = total_train_loss / len(self.train_loader)
                history["train_loss"].append(avg_train_loss)
                print(f"Epoch {epoch}/{num_epochs} → Avg Train Loss: {avg_train_loss:.6f}")

                # Validation
                self.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                        batch = batch.to(device)
                        batch = self.forward(batch)

                        lc, le = 0.0, 0.0
                        for i in range(batch.num_nodes):
                            lc += F.mse_loss(batch.pred_center_blocks[i], batch.target_center_blocks[i].to(device))
                        for k in range(batch.num_edges):
                            le += F.mse_loss(batch.pred_edge_blocks[k], batch.target_edge_blocks[k].to(device))
                        total_val_loss += ((lc + le) / batch.num_graphs).item()

                avg_val_loss = total_val_loss / len(self.val_loader)
                history["val_loss"].append(avg_val_loss)
                print(f"Epoch {epoch}/{num_epochs} → Avg Val   Loss: {avg_val_loss:.6f}")

                # early stop!
                if avg_val_loss < best_val: 
                    best_val = avg_val_loss
                    no_imp_epochs = 0
                    if model_save_path: 
                        self.save_model_checkpoint(model_save_path, epoch, optimizer)
                else:
                    no_imp_epochs += 1
                    if no_imp_epochs >= grace_epochs: 
                        print(f"No improvement for {grace_epochs} -> early stopping")
                        break
                scheduler.step(avg_val_loss)
                history["lr"].append(optimizer.param_groups[0]['lr'])
        except KeyboardInterrupt:
            print("Training interrupted by user. Benchmark...")

        

        # test performance: 
        self.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Epoch {epoch} [Test]"):
                batch = batch.to(device)
                batch = self.forward(batch)

                lt, le = 0.0, 0.0
                for i in range(batch.num_nodes):
                    lt += F.mse_loss(batch.pred_center_blocks[i], batch.target_center_blocks[i].to(device))
                for k in range(batch.num_edges):
                    le += F.mse_loss(batch.pred_edge_blocks[k], batch.target_edge_blocks[k].to(device))
                total_test_loss += ((lt + le) / batch.num_graphs).item()
        avg_test_loss = total_test_loss / len(self.test_loader)
        history["test_loss"] = avg_test_loss
        # save history
        import pickle
        base, _ = os.path.splitext(model_save_path)
        hist_path = base + ".history"
        with open(hist_path, "wb") as f: 
            pickle.dump(history, f)
        print(f"Test  Loss: {avg_test_loss:.6f}")
        # already saved above!
        # if model_save_path is not None:
        #     self.save_model_checkpoint(model_save_path, epoch, optimizer)

    def save_model(self, path):
        """Save the model to the specified path."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
        }
        torch.save(checkpoint, path)
        dprint(1, f"Model saved to {path}")
    
    def save_model_checkpoint(self, path, epoch, optimizer=None):
        """Save the model checkpoint to the specified path."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, path)
        dprint(1, f"Model checkpoint saved to {path}")
        
    def load_model(self, path: str, strict: bool = True):
        """
        Load weights from `path` into this model. 
        If `strict=True`, will error if keys don't match exactly.
        If `strict=False`, will load only matching keys and ignore others.
        """
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        try:
            self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            print(f"Loaded weights from {path} (strict={strict})")
        except RuntimeError as e:
            if strict:
                print("Compatibility check failed, retrying with strict=False…")
                self.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print(f"Loaded partial weights from {path} (strict=False)")
            else:
                raise e

if __name__ == "__main__": 
    BASIS_PATH = "scripts/6-31g_2df_p_custom_nwchem.gbs"
    GEOMETRY_Source = "datasets/QM9/xyz_c7h10o2_sorted"
    MGNN = MolGraphNetwork(xyz_source=GEOMETRY_Source, backend=Backend.PY, basis=BASIS_PATH, batch_size=2)
    MGNN.load_data(max_samples=10, 
                   cache_meta={"method":"dft", "basis":None, "functional": "b3lypg", "guess": "minao", "backend": "pyscf", "cache": "datasets/QM9/out/c7h10o2_b3lypg_6-31G(2df,p)/pyscf"})
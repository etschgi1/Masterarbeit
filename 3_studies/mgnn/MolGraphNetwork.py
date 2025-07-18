import os, copy, tempfile
from typing import List
from itertools import chain
from scf_guess_tools import Backend, load
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from collections import defaultdict
from torch_scatter import scatter_add #! maybe use other aggregation functions later on

from mgnn.Graphutils import dprint, set_verbose, density_fock_overlap, unflatten_triang, rotate_points, rotate_M, rotated_xyz_content

from mgnn.encoder import EncoderDecoderFactory


set_verbose(2)  # Set the verbosity level for debugging output

## Defines
ATOM_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


class MolGraphNetwork(torch.nn.Module): 
    """A class to controll the GNN for density matrix prediction."""

    def __init__(self, 
                 dataset,
                 basis, 
                 max_block_dim=26,
                 hidden_dim=256,
                 batch_size=32,
                 message_passing_steps=2, # Number of message passing steps
                 backend=Backend.PY,
                 edge_threshold_type="atom_dist",
                 edge_threshold_val=3, # Angrom for "atom_dist", or dimensionless for "max" or "mean" 
                 target="density", # "fock" or "density"
                 data_aug_factor=1, # 1==no augmentation
                 **kwargs
                 ):
        super().__init__()

        self.dataset = dataset
        self.backend = backend
        self.basis = basis
        self.edge_threshold_type = edge_threshold_type  # Can be "max" or "mean" to determine the threshold for edges
        assert edge_threshold_type in ["atom_dist", "max", "mean", "fro"], "edge_threshold_type must be 'max' or 'mean'."
        self.edge_threshold = edge_threshold_val  # Default threshold for edge creation
        assert edge_threshold_val > 0, "edge_threshold must be a positive value."
        assert target in ["fock", "density"], "target must be either 'fock' or 'density'."
        self.target = target
        self.data_aug_factor = data_aug_factor 

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
        self.cur_val_loss = None

        # Encoder / Decoder factory to generate stuff for different atom types - instantiated in setup_model inside load_data
        self.node_encoders = None # encodes upper triangular blocks for center blocks
        self.node_updaters = None # updates node features based on messages
        self.center_decoders = None # decodes hidden node features to upper triangular blocks for centers
        self.edge_encoders = None # encodes hetero/homo blocks for edges
        self.edge_decoders = None # decodes hidden edge features to hetero/homo blocks for edges
        self.message_net = None # Net to provide message passing! 

        self.no_progress_bar = False

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if hasattr(self, "verbose_level"): 
            print(f"Setting verbose level to: {self.verbose_level}")
            set_verbose(self.verbose_level)


    def load_data(self):
        """Load data from source directory split into train and test sets and create normalized BlockMatrices."""
        self.molgraphs = [] #reset in case this is called again later!
        dprint(1, f"Loading {self.dataset.size} files from {self.dataset.name}...")
        
        xyz_root = self.dataset.xyz
        def load_set(set_name: str): 
            focks_in, dens_in, overlap_in, coords_in, files_in = [], [], [], [], []
            set_keys = self.dataset.train_keys if set_name == "train" else self.dataset.val_keys if set_name == "val" else self.dataset.test_keys
            dprint(2, f"Loading {len(set_keys)} files for {set_name} set from {xyz_root}...")
            for key in set_keys:
                result = self.dataset.solution(key)
                dens_in.append(result.density)
                focks_in.append(result.fock)
                overlap_in.append(result.overlap)
                xyz_file = os.path.join(xyz_root, self.dataset.names[key] + ".xyz")

                with open(xyz_file, 'r') as f:
                    lines = f.readlines()
                    coords_count = int(lines[0])  # First line contains the number of atoms
                    coords = [list(map(float, line.split()[1:4])) for line in lines[2:coords_count+2]]
                    assert len(coords) == int(lines[0]), f"Number of coordinates {len(coords)} does not match number of atoms {int(lines[0])} in file {xyz_file}"
                    coords_in.append(coords)
                    files_in.append(xyz_file)
            if self.target == "fock":
                return focks_in, overlap_in, coords_in, files_in
            elif self.target == "density":
                return dens_in, overlap_in, coords_in, files_in
            else:
                raise ValueError(f"Unknown target {self.target}. Must be 'fock' or 'density'.")

        train_data = load_set("train")
        val_data = load_set("val")
        test_data = load_set("test")
        train_size = len(train_data[0])
        val_size = len(val_data[0])
        test_size = len(test_data[0])

        #! gather train graphs + augment data if needed
        self.train_graphs = [] 
        for key, target, overlap, coords, xyz_file in tqdm(zip(self.dataset.train_keys, *train_data), desc="Creating training graphs", disable=self.no_progress_bar):
            mol = self.dataset.molecule(key)
            self.train_graphs.append(self.make_graph(overlap, target, coords, mol))
        n_aug = 0
        if self.data_aug_factor > 1:
            target_in, overlap_in, coords_in, files_in = train_data
            n_aug = int(self.data_aug_factor * train_size - train_size)
            aug_graphs, aug_target_in, aug_overlap_in, aug_infos, aug_coords = [], [], [], [], []
            dprint(1, f"Augmenting training set using factor {self.data_aug_factor} -> {n_aug} additional training samples.")
            for _ in tqdm(range(int(n_aug)), desc="Augmenting data", disable=self.no_progress_bar):
                idx = np.random.choice(range(len(train_data))) #! only augment training data
                overlap, target, coords, xyz_file = overlap_in[idx], target_in[idx], coords_in[idx], files_in[idx]
                aug_graph, aug_overlap, aug_target, aug_info = self.aug_data(overlap, target, coords, xyz_file)
                aug_graphs.append(aug_graph)
                aug_overlap_in.append(aug_overlap)
                aug_target_in.append(aug_target)
                aug_infos.append(aug_info)  
                aug_coords.append(coords)  # keep the coordinates for the augmented data
            self.train_graphs.extend(aug_graphs)
            overlap_in.extend(aug_overlap_in)
            target_in.extend(aug_target_in)
            coords_in.extend(aug_coords) # same as non-augmented!
            files_in.extend(aug_infos)  # augmentation info instead of filenames (this includes filenames)
        total_samples = train_size + val_size + test_size + n_aug

        # validation and test data
        self.val_graphs = []
        self.test_graphs = []
        for key, target, overlap, coords, xyz_file in tqdm(zip(self.dataset.val_keys, *val_data), desc="Creating validation graphs", disable=self.no_progress_bar):
            mol = self.dataset.molecule(key)
            self.val_graphs.append(self.make_graph(overlap, target, coords, mol))
        for key, target, overlap, coords, xyz_file in tqdm(zip(self.dataset.test_keys, *test_data), desc="Creating test graphs", disable=self.no_progress_bar):
            mol = self.dataset.molecule(key)
            self.test_graphs.append(self.make_graph(overlap, target, coords, mol))

       
        self.test_ground_truth = [target for target in test_data[0]]  
        self.val_ground_truth = [target for target in val_data[0]]
        self.train_ground_truth = [target for target in train_data[0]] 
        self.test_ovlp_mat = [overlap for overlap in test_data[1]]
        self.val_ovlp_mat = [overlap for overlap in val_data[1]]
        self.train_ovlp_mat = [overlap for overlap in train_data[1]]
        self.files = {"train": [xyz_f for xyz_f in train_data[3]],
                      "val": [xyz_f for xyz_f in val_data[3]],
                      "test": [xyz_f for xyz_f in test_data[3]]}
        
        if self.data_aug_factor > 1:
            dprint(1, f"Total samples: {total_samples}, Train: {train_size} (with {n_aug} / {train_size} augmented samples), Val: {val_size}, Test: {test_size}")
        else:
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

    def aug_data(self, overlap, target, coords, xyz_file): 
        mol = load(xyz_file, backend=Backend.PY, basis=self.basis).native
        rand_axis = np.random.normal(size=(3,))
        rand_axis /= np.linalg.norm(rand_axis)
        rand_angle = np.random.uniform(0, 2 * np.pi)
        rotated_overlap = rotate_M(mol, rand_axis, rand_angle, overlap)
        rotated_target = rotate_M(mol, rand_axis, rand_angle, target)
        rotated_coords = rotate_points(coords, rand_axis, rand_angle)
        
        #! own temp file for every process to avoid race conditions!
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as tf:
            tf.writelines(rotated_xyz_content(xyz_file, rotated_coords))
            tmp_path = tf.name
        try:
            rotated_mol = load(tmp_path, backend=Backend.PY, basis=self.basis).native
        except Exception as e:
            dprint(0, f"Error loading rotated molecule from {tmp_path}: {e}")
            dprint(0, xyz_file)
            dprint(0, rotated_coords)
            with open(tmp_path, 'r') as f:
                dprint(0, f"Rotated coordinates content:\n{f.read()}")
            raise e
        finally: 
            os.remove(tmp_path) # clean up manually
        aug_graph = self.make_graph(rotated_overlap, rotated_target, rotated_coords, rotated_mol)
        return aug_graph, rotated_overlap, rotated_target, (xyz_file, rand_axis, rand_angle)  

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

       # --> offset and collect ao_slices and edge_ao_slices needed for loss_on_full_matrix=True
        cum = 0
        all_ao, all_edge_ao = [], []

        for g in batch:
            # atom-slices: each is (field1, field2, start, end)
            for (f1, f2, s, e) in g.ao_slices:
                all_ao.append((f1, f2, s + cum, e + cum))

            # edge-slices are already pairs of 2-tuples (start,end)
            for ((si, ei), (sj, ej)) in g.edge_ao_slices:
                all_edge_ao.append(
                    ((si + cum, ei + cum),
                    (sj + cum, ej + cum))
                )

            cum += g.num_nodes

        batch_data.ao_slices      = all_ao
        batch_data.edge_ao_slices = all_edge_ao

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
    
    def rebuild_full_torch(self, pred_centers, pred_edges, ao_slices, edge_ao_slices, N, device):
        out = torch.zeros(N, N, device=device)
        """This supports backpropagation through the full matrix reconstruction - not needed for prediction!"""
        for i, (_, _, s, e) in enumerate(ao_slices):
            d = e - s
            flat = pred_centers[i]                  # shape [d*(d+1)//2]
            idx = torch.triu_indices(d, d, 0, device=device)
            block = torch.zeros(d, d, device=device)
            block[idx[0], idx[1]] = flat
            # mirror lower triangle
            block = block + block.T - torch.diag(block.diag())
            out[s:e, s:e] = block
        # edges
        for i, ((si, ei), (sj, ej)) in enumerate(edge_ao_slices):
            flat = pred_edges[2*i]                  # directed‐edge doubling
            block = flat.view(ei - si, ej - sj)
            out[si:ei, sj:ej] = block
            out[sj:ej, si:ei] = block.T
        return out
    
    def rebuild_matrix(self, pred_center_blocks, pred_edge_blocks, ao_slices, edge_ao_slices):
        N = sum(end-start for _, _, start, end in ao_slices)  
        out = np.zeros((N, N), dtype=np.float64)  
        # place center blocks
        for i, (_, _, start, end) in enumerate(ao_slices):
            flat_center_block = pred_center_blocks[i]
            out[start:end, start:end] = unflatten_triang(flat_center_block.cpu().numpy(), end - start)
        for i, ((start_i, end_i), (start_j, end_j)) in enumerate(edge_ao_slices):
            flat_edge_block = pred_edge_blocks[2*i].cpu() #! OC we have to index every second because we doubled the edges (directed edges - see make_graph) but not the indices! 
            block = flat_edge_block.reshape(end_i - start_i, end_j - start_j)  
            out[start_i:end_i, start_j:end_j] = block
            out[start_j:end_j, start_i:end_i] = block.T
        return out
        
    def predict(self, graphs: List[Data], inv_transform=True, raw=False, include_target=False, transform_to_density=True): 
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
                density matrices (Note that only one spin component is given *2 to get alpha & beta).
                Default: True.

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
                        # print(f"Ovlp: {ovlp[:10, :10]}")
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
            raw_center_blocks = torch.stack([batch.center_blocks[i].to(device) for i in atom_sym_indices], dim=0) # This now has shape (Nr_of_atoms_for_this_sym, self.center_sizes[u_sym])
            assert raw_center_blocks.shape[1] == self.center_sizes[u_sym], f"Center block size {raw_center_blocks.shape[1]} does not match expected size {self.center_sizes[u_sym]} for atom type {u_sym}."
            try:
                c_sym = self.node_encoders[u_sym](raw_center_blocks) 
            except KeyError as excep:
                # no encoder available -> set 0
                c_sym = torch.zeros((len(atom_sym_indices), self.hidden_dim), device=device)  # No encoder for this atom type, so we set it to zero
            c[atom_sym_indices] = c_sym  # in h we have the same dimensiom self.hidden_dim for all atoms in the batch

        # II) Encode edge features (edge blocks)
        unique_edge_keys = set(batch.edge_pair_sym)  # Unique edge types
        edge_indices_dict = defaultdict(list)  
        e = torch.zeros((E_total, self.hidden_dim), device=device)  # Edge features
        for key in unique_edge_keys:
            edge_key_indices = [i for i, sym in enumerate(batch.edge_pair_sym) if sym == key]
            edge_indices_dict[key].extend(edge_key_indices)  # Store indices for this edge type
            raw_edge_blocks = torch.stack([batch.edge_blocks[i].to(device) for i in edge_key_indices], dim=0)
            distances = batch.edge_dist[edge_key_indices].to(device).view(-1, 1)  # Reshape distances to match edge blocks -> (Nr of edges for this key, 1)
            edge_inputs = torch.cat((raw_edge_blocks, distances), dim=1)  
            try:
                e_key = self.edge_encoders[key](edge_inputs)  
            except KeyError as excep:
                # no encoder available -> set 0
                e_key = torch.zeros((len(edge_key_indices), self.hidden_dim), device=device)
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
                try:
                    c_new[i] = self.node_updaters[sym](old_and_agg)  # Update node features with the aggregated messages
                except KeyError as excep:
                    # no updater available -> set 0
                    c_new[i] = torch.zeros(self.hidden_dim, device=device)
            c = c_new 
        
        # IV) Decode node features to center blocks
        pred_center_blocks = [None] * len(batch.center_blocks)  # Note that we do not use numpy arrays here because differnt blocks have different sizes!
        for sym in unique_atom_syms:
            atom_sym_indices = atom_indices_dict[sym] #reuse the indices from encoding
            c_sym_stack = torch.stack([c[i] for i in atom_sym_indices], dim=0)  # (Nr_of_atoms_for_this_sym, self.hidden_dim)
            try:
                center_decoded = self.center_decoders[sym](c_sym_stack)  # Decode to center blocks
            except KeyError as excep:
                center_decoded = torch.zeros((len(atom_sym_indices), self.center_sizes[sym]), device=device)  # No decoder for this atom type, so we set it to zero
            for i, idx in enumerate(atom_sym_indices):
                pred_center_blocks[idx] = center_decoded[i]
        
        # V) Decode edge features to edge blocks
        pred_edge_blocks = [None] * len(batch.edge_blocks) 
        for key in unique_edge_keys: 
            edge_key_indices = edge_indices_dict[key]
            e_key_stack = torch.stack([e[i] for i in edge_key_indices], dim=0)  # (Nr_of_edges_for_this_key, self.hidden_dim)
            try:
                edge_decoded = self.edge_decoders[key](e_key_stack)  
            except KeyError as excep:
                edge_decoded = torch.zeros((len(edge_key_indices), self.edge_sizes[key]), device=device)  # No decoder for this edge type, so we set it to zero
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
        def add_types(graphs):
            found = 0
            for graph in graphs:
                for i, atom_sym in enumerate(graph.atom_sym):
                    if atom_sym not in self.center_sizes:
                        self.center_sizes[atom_sym] = graph.center_blocks[i].shape[0]
                        dprint(2, f"Found center block size {self.center_sizes[atom_sym]} for atom type {atom_sym}.")
                        found += 1
                for i, edge_sym in enumerate(graph.edge_pair_sym):
                    if edge_sym not in self.edge_sizes:
                        self.edge_sizes[edge_sym] = graph.edge_blocks[i].shape[0]
                        dprint(2, f"Found edge block size {self.edge_sizes[edge_sym]} for edge type {edge_sym}.")
                        found += 1
            return found
        add_types(self.train_graphs)  # gather from training graphs
        # additionally setup stats for val and test graphs
        found_val = add_types(self.val_graphs)
        found_test = add_types(self.test_graphs)
        if found_val > 0 or found_test > 0:
            dprint(1, "===Gathering block size statistics from validation and test graphs===")
            dprint(1, f"!!!Found {found_val} new center/edge block sizes in validation graphs and {found_test} in test graphs.")
            dprint(1, f"NO PREDICTIONS WILL BE MADE ON CENTER / EDGE BLOCKS OF THESE TYPES! Please ensure that all types are present in the training set.")

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
                try:
                    # source
                    S_mean_t, S_std_t = torch.tensor(self.center_norm[key][0], device=S_center_block.device), torch.tensor(self.center_norm[key][1], device=S_center_block.device) # to use device block is stored on
                    graph.center_blocks[i] = (S_center_block - S_mean_t) / S_std_t
                    # target
                    T_mean_t, T_std_t = torch.tensor(self.center_norm_target[key][0], device=T_center_block.device), torch.tensor(self.center_norm_target[key][1], device=T_center_block.device)
                    graph.target_center_blocks[i] = (T_center_block - T_mean_t) / T_std_t
                except KeyError as e:
                    continue  # If the key is not found in the normalization dict, skip for this atom type
            # edges
            for i, (S_edge_block, T_edge_block) in enumerate(zip(graph.edge_blocks, graph.target_edge_blocks)):
                key = graph.edge_pair_sym[i]
                # source
                try:  # If the key is not found in the normalization dict, skip for this edge type
                    S_mean_t, S_std_t = torch.tensor(self.edge_norm[key][0], device=S_edge_block.device), torch.tensor(self.edge_norm[key][1], device=S_edge_block.device)
                    graph.edge_blocks[i] = (S_edge_block - S_mean_t) / S_std_t
                    # target
                    T_mean_t, T_std_t = torch.tensor(self.edge_norm_target[key][0], device=T_edge_block.device), torch.tensor(self.edge_norm_target[key][1], device=T_edge_block.device)
                    graph.target_edge_blocks[i] = (T_edge_block - T_mean_t) / T_std_t
                except KeyError as e:
                    continue # same for edges
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
                try:
                    S_mean_t, S_std_t = torch.tensor(self.center_norm[key][0], device=S_center_block.device), torch.tensor(self.center_norm[key][1], device=S_center_block.device)
                    inv_g.center_blocks[i] = S_center_block * S_std_t + S_mean_t

                    T_mean_t, T_std_t = torch.tensor(self.center_norm_target[key][0], device=T_center_block.device), torch.tensor(self.center_norm_target[key][1], device=T_center_block.device)
                    inv_g.target_center_blocks[i] = T_center_block * T_std_t + T_mean_t

                    inv_g.pred_center_blocks[i] = P_center_block * T_std_t + T_mean_t  # Uses same stats as target (from training set)
                except KeyError as e:
                    continue
            # edges
            for i, (S_edge_block, T_edge_block, P_edge_block) in enumerate(zip(inv_g.edge_blocks, inv_g.target_edge_blocks, inv_g.pred_edge_blocks)):
                key = inv_g.edge_pair_sym[i]
                try:
                # source
                    S_mean_t, S_std_t = torch.tensor(self.edge_norm[key][0], device=S_edge_block.device), torch.tensor(self.edge_norm[key][1], device=S_edge_block.device)
                    inv_g.edge_blocks[i] = S_edge_block * S_std_t + S_mean_t
                    # target
                    T_mean_t, T_std_t = torch.tensor(self.edge_norm_target[key][0], device=T_edge_block.device), torch.tensor(self.edge_norm_target[key][1], device=T_edge_block.device)
                    inv_g.target_edge_blocks[i] = T_edge_block * T_std_t + T_mean_t

                    inv_g.pred_edge_blocks[i] = P_edge_block * T_std_t + T_mean_t  # Uses same stats as target (from training set)
                except KeyError as e:
                    continue

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
    
    def train_model(self, num_epochs=5, lr=1e-3, weight_decay=1e-5, device=None, model_save_path=None, grace_epochs=3, 
                    lr_args={"mode":"min",
                             "factor": 0.5,
                             "patience": 3,
                             "threshold": 1e-3,
                             "cooldown": 2, 
                             "min_lr" : 1e-6}, 
                    report_fn=None, loss_on_full_matrix=False, cpu_training=False):
        """Train the GNN on the training set, validate & test, with optional early stopping and LR scheduling.

        Args:
            num_epochs (int): maximum number of training epochs.
            lr (float): initial learning rate for AdamW optimizer.
            weight_decay (float): weight decay (L2 penalty).
            device (torch.device or str): device to run training on (e.g. 'cuda' or 'cpu').
            model_save_path (str or None): path to save best model checkpoints; if None, no checkpoints are saved.
            grace_epochs (int): number of epochs with no improvement on val loss before early stopping.
            lr_args (dict): arguments for ReduceLROnPlateau scheduler:
                - mode (str): 'min' or 'max'
                - factor (float): LR reduction factor
                - patience (int): epochs to wait before reducing LR
                - threshold (float): threshold for measuring improvement
                - cooldown (int): epochs to wait after LR reduction
                - min_lr (float): minimum LR
            report_fn (callable or None): function to report intermediate metrics (e.g. to Ray Tune); called as report_fn({"loss": val_loss, "epoch": epoch, ...}).
            loss_on_full_matrix (bool): if True, compute loss on reconstructed full matrices (MSE/RMSE) instead of block-wise loss.

        Returns:
            None  (prints training/validation/test loss and saves history/checkpoints if configured)
        """
        import torch.nn.functional as F
        from tqdm import tqdm
        setattr(self, "model_save_path", model_save_path)  # save path for later use
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cpu_training:
            device = torch.device("cpu")
            dprint(1, "Training on CPU! Overwrite is set!")
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode= lr_args["mode"],
                                                               factor= lr_args["factor"],
                                                               patience= lr_args["patience"],
                                                               threshold= lr_args["threshold"],
                                                               cooldown= lr_args["cooldown"],
                                                               min_lr= lr_args["min_lr"])
        history = {
            "train_loss": [],
            "val_loss":[],
            "test_loss":float,
            "lr":[],
        }
        best_val, no_imp_epochs = float('inf'), 0

        def compute_loss(batch, device):
            if loss_on_full_matrix: 
                max_dim = max(e for (_,_,_,e) in batch.ao_slices)
                rebuilded_matrices = self.rebuild_full_torch(batch.pred_center_blocks, batch.pred_edge_blocks, batch.ao_slices, batch.edge_ao_slices, N=max_dim, device=device)
                target_matrices = self.rebuild_full_torch(batch.target_center_blocks, batch.target_edge_blocks, batch.ao_slices, batch.edge_ao_slices, N=max_dim, device=device)
                return F.mse_loss(rebuilded_matrices.to(device), target_matrices.to(device), reduction="sum") / batch.num_graphs 
            else:
                loss_center, loss_edge = 0.0, 0.0
                for i in range(batch.num_nodes):
                    loss_center += F.mse_loss(batch.pred_center_blocks[i], batch.target_center_blocks[i].to(device))
                for k in range(batch.num_edges):
                    loss_edge += F.mse_loss(batch.pred_edge_blocks[k], batch.target_edge_blocks[k].to(device))
                return (loss_center + loss_edge) / batch.num_graphs

        try: 
            for epoch in range(1, num_epochs + 1):
                self.train()
                total_train_loss = 0.0

                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", disable=self.no_progress_bar):
                    ao_slices, edge_ao_slices = batch.ao_slices, batch.edge_ao_slices
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    batch.ao_slices = ao_slices
                    batch.edge_ao_slices = edge_ao_slices
                    batch = self.forward(batch)
                    # loss_center, loss_edge = 0.0, 0.0
                    # for i in range(batch.num_nodes):
                    #     loss_center += F.mse_loss(batch.pred_center_blocks[i], batch.target_center_blocks[i].to(device))
                    # for k in range(batch.num_edges):
                    #     loss_edge += F.mse_loss(batch.pred_edge_blocks[k], batch.target_edge_blocks[k].to(device))

                    # loss = (loss_center + loss_edge) / batch.num_graphs
                    loss = compute_loss(batch, device)
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
                    for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", disable=self.no_progress_bar):
                        batch = batch.to(device)
                        batch = self.forward(batch)

                        # lc, le = 0.0, 0.0
                        # for i in range(batch.num_nodes):
                        #     lc += F.mse_loss(batch.pred_center_blocks[i], batch.target_center_blocks[i].to(device))
                        # for k in range(batch.num_edges):
                        #     le += F.mse_loss(batch.pred_edge_blocks[k], batch.target_edge_blocks[k].to(device))
                        # total_val_loss += ((lc + le) / batch.num_graphs).item()
                        total_val_loss += compute_loss(batch, device).item()

                avg_val_loss = total_val_loss / len(self.val_loader)
                if report_fn is not None: # report to ray!
                    report_fn({"loss": avg_val_loss, "epoch": epoch, "train_loss": avg_train_loss})

                history["val_loss"].append(avg_val_loss)
                print(f"Epoch {epoch}/{num_epochs} → Avg Val   Loss: {avg_val_loss:.6f}")
                self.cur_val_loss = avg_val_loss 
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
            print("Training interrupted by user. Benchmark model...")

        

        # test performance: 
        self.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Epoch {epoch} [Test]", disable=self.no_progress_bar):
                batch = batch.to(device)
                batch = self.forward(batch)

                # lt, le = 0.0, 0.0
                # for i in range(batch.num_nodes):
                #     lt += F.mse_loss(batch.pred_center_blocks[i], batch.target_center_blocks[i].to(device))
                # for k in range(batch.num_edges):
                #     le += F.mse_loss(batch.pred_edge_blocks[k], batch.target_edge_blocks[k].to(device))
                # total_test_loss += ((lt + le) / batch.num_graphs).item()
                total_test_loss += compute_loss(batch, device).item()
        avg_test_loss = total_test_loss / len(self.test_loader)
        history["test_loss"] = avg_test_loss
        if model_save_path:
            # save history
            import pickle
            base, _ = os.path.splitext(model_save_path)
            hist_path = base + ".history"
            with open(hist_path, "wb") as f: 
                pickle.dump(history, f)
        print(f"Test  Loss: {avg_test_loss:.6f}")
        return epoch, history

    def save_model(self, path, epoch=0):
        """Save the model to the specified path."""
        checkpoint = {
            'epoch': epoch, 
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
    
    def get_last_epoch(self, save_path=None): 
        save_path = getattr(self, 'model_save_path', save_path) 
        checkpoint = torch.load(save_path, map_location=lambda storage, loc: storage)
        if "epoch" in checkpoint:
            return checkpoint["epoch"]
        else:
            print(f"No epoch information found in checkpoint {save_path}. Returning 0.")
            return 0
        
    def load_model(self, path: str, strict: bool = True):
        """
        Load weights from `path` into this model. 
        If `strict=True`, will error if keys don't match exactly.
        If `strict=False`, will load only matching keys and ignore others.
        """
        self.model_save_path = path  # Store the path for later use
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
    print("Import test for MolGraphNetwork.py")
    print("This file is not meant to be run directly. Use it as a module in your training script.")
import itertools
import torch
from typing import List, Dict
from mgnn.message_net import MessageNet


class EncoderDecoderFactory(torch.nn.Module):
    """
    Builds node- and edge-encoder/decoder ModuleDicts for arbitrary atom types.

    - atom_types: a list of element symbols, e.g. ["C","H","O"]
    - hidden_dim: dimension of every hidden embedding
    - max_up: size of the largest possible center-block's flattened upper-triangle
    - max_sq: size of the largest possible hetero-block's flattened full matrix

    Produces:
      • node_encoders:    ModuleDict mapping each atom symbol → MLP(max_up→hidden_dim)
      • node_updaters:    ModuleDict mapping each atom symbol → MLP(2*hidden_dim→hidden_dim)
      • center_decoders:  ModuleDict mapping each atom symbol → Linear(hidden_dim→max_up)
      • edge_encoders:    ModuleDict mapping each pair-key → MLP(max_sq+1→hidden_dim)
      • edge_decoders:    ModuleDict mapping each pair-key → Linear(hidden_dim→max_sq)

    A “pair-key” is the sorted join of two symbols, e.g. "C_H", "H_H", "C_O", etc.
    """

    def __init__(self, 
                 atom_types: List[str],
                 edge_types: List[str],
                 hidden_dim: int,
                 center_sizes: Dict[str, int],
                 edge_sizes: Dict[str, int],
                 message_layers: int = 3, 
                 message_dropout: float = 0.1):
        """atom_types: list of element symbols, e.g. ["C","H","O"]
            edge_types: list of edge types, e.g. ["C_H", "H_H", "C_O", "H_O"]
            hidden_dim: dimension of every hidden embedding
            center_sizes: dict giving the sizes of the center for a given atom type
            edge_sizes: dict giving the sizes of the edge for a given edge_type
            message_layers: number of layers in the message net (default=3)
            message_dropout: dropout probability in the message net (default=0.1) - can also be list of floats for each layer
        """
        
        super().__init__()
        self.atom_types = atom_types
        self.hidden_dim = hidden_dim
        self.center_sizes = center_sizes
        self.edge_sizes = edge_sizes

        # 1) NODE ENCODERS (center-block → hidden)
        self.node_encoders = torch.nn.ModuleDict({
            sym: torch.nn.Sequential(
                torch.nn.Linear(self.center_sizes[sym], hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            for sym in atom_types
        })

        # 2) NODE UPDATERS (hidden+agg_hidden → new_hidden)
        #    [h_i || sum_messages_i] → updated h_i
        self.node_updaters = torch.nn.ModuleDict({
            sym: torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            for sym in atom_types
        })

        # 3) NODE DECODERS (hidden → flattened center-block)
        self.center_decoders = torch.nn.ModuleDict({
            sym: torch.nn.Linear(hidden_dim, self.center_sizes[sym])
            for sym in atom_types
        })

        # 4) EDGE ENCODERS (flattened hetero/homo-block + dist → hidden) + 1 for distance! 
        #    Build keys for all unordered pairs of atom_types (including same-element for homo blocks)
        self.edge_encoders = torch.nn.ModuleDict({
            key: torch.nn.Sequential(
                torch.nn.Linear(self.edge_sizes[key] + 1, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            for key in edge_types
        })

        # 5) EDGE DECODERS (hidden → flattened hetero/homo-block)
        self.edge_decoders = torch.nn.ModuleDict({
            key: torch.nn.Linear(hidden_dim, self.edge_sizes[key])
            for key in edge_types
        })

        # 6) MESSAGE NET (shared MLP for combining [h_i, h_j, edge_emb])
        input_dim = 3 * hidden_dim  # h_i, h_j, edge_emb
        self.message_net = MessageNet(input_dim = input_dim,
                                      hidden_dim = hidden_dim,
                                      num_layers = message_layers,
                                      dropout = message_dropout)

    @staticmethod
    def _make_all_pair_keys(atom_types):
        """
        Return all sorted unordered pair-keys for the given atom_types.
        E.g. atom_types = ["C","H","O"] → ["C_C","C_H","C_O","H_H","H_O","O_O"].
        """
        pairs = itertools.combinations_with_replacement(sorted(atom_types), 2)
        return [f"{a}_{b}" for a, b in pairs]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use factory.node_encoders[...] etc., not forward().")

import itertools
import torch
from message_net import MessageNet


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

    def __init__(self, atom_types, hidden_dim, max_up, max_sq, message_layers=2, 
                 message_dropout=2):
        """atom_types: list of element symbols, e.g. ["C","H","O"]
            hidden_dim: dimension of every hidden embedding
            max_up: size of the largest possible center-block's flattened upper-triangle
            max_sq: size of the largest possible hetero-block's flattened full matrix
            message_layers: number of layers in the message net (default=2)
            message_dropout: dropout probability in the message net (default=0.0)
        """
        
        super().__init__()
        self.atom_types = atom_types
        self.hidden_dim = hidden_dim
        self.max_up = max_up
        self.max_sq = max_sq

        # 1) NODE ENCODERS (center-block → hidden)
        self.node_encoders = torch.nn.ModuleDict({
            sym: torch.nn.Sequential(
                torch.nn.Linear(max_up, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            for sym in atom_types
        })

        # 2) NODE UPDATERS (hidden+agg_hidden → new_hidden)
        #    [h_i || sum_messages_i] → updated h_i
        self.node_updaters = torch.nn.ModuleDict({
            sym: torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            for sym in atom_types
        })

        # 3) CENTER DECODERS (hidden → flattened center-block)
        self.center_decoders = torch.nn.ModuleDict({
            sym: torch.nn.Linear(hidden_dim, max_up)
            for sym in atom_types
        })

        # 4) EDGE ENCODERS (flattened hetero/homo-block + dist → hidden)
        #    Build keys for all unordered pairs of atom_types (including same-element for homo blocks)
        edge_keys = self._make_all_pair_keys(atom_types)
        self.edge_encoders = torch.nn.ModuleDict({
            key: torch.nn.Sequential(
                torch.nn.Linear(max_sq + 1, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            for key in edge_keys
        })

        # 5) EDGE DECODERS (hidden → flattened hetero/homo-block)
        self.edge_decoders = torch.nn.ModuleDict({
            key: torch.nn.Linear(hidden_dim, max_sq)
            for key in edge_keys
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
        """
        This class is not meant to be called directly in forward. Instead,
        instantiate it and then use its attributes:
          - factory.node_encoders[sym]
          - factory.edge_encoders[pair_key]
          - etc.
        """
        raise NotImplementedError("Use factory.node_encoders[...] etc., not forward().")

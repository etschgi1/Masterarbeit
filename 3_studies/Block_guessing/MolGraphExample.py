import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def create_example_data():
    """
    Create a small example Data object that mimics the structure produced by make_graph().
    - 3 atoms: C, H, O
    - Center-blocks: C is 3x3, H is 2x2, O is 4x4 (we only store their upper-triangles)
    - Edges: C-H, H-O, C-O (each undirected → 2 directed edges)
    """
    # 1) Node-level: atom symbols and “center” blocks (upper-triangle only)
    atom_syms = ["C", "H", "O"]

    # Build small blocks just for illustration:
    C_mat = np.arange(9).reshape(3, 3)   # pretend overlap block for C (3x3)
    H_mat = np.arange(4).reshape(2, 2)   # pretend overlap block for H (2x2)
    O_mat = np.arange(16).reshape(4, 4)  # pretend overlap block for O (4x4)

    # Extract upper-triangles:
    C_center = torch.tensor(np.triu(C_mat)[np.triu_indices(3)], dtype=torch.float)
    H_center = torch.tensor(np.triu(H_mat)[np.triu_indices(2)], dtype=torch.float)
    O_center = torch.tensor(np.triu(O_mat)[np.triu_indices(4)], dtype=torch.float)

    center_blocks = [C_center, H_center, O_center]

    # 2) Edge-level: define three undirected pairs → 6 directed edges
    #    C-H block: 3x2  (flattened length 6)
    #    H-O block: 2x4  (flattened length 8)
    #    C-O block: 3x4  (flattened length 12)
    CH_mat = np.arange(6).reshape(3, 2)
    HO_mat = np.arange(8).reshape(2, 4)
    CO_mat = np.arange(12).reshape(3, 4)

    CH_block = torch.tensor(CH_mat.reshape(-1), dtype=torch.float)
    HO_block = torch.tensor(HO_mat.reshape(-1), dtype=torch.float)
    CO_block = torch.tensor(CO_mat.reshape(-1), dtype=torch.float)

    CH_dist = 1.2
    HO_dist = 1.5
    CO_dist = 1.4

    CH_sym = "C_H"
    HO_sym = "H_O"
    CO_sym = "C_O"

    edge_index_list = []
    edge_blocks = []
    edge_dists = []
    edge_syms = []

    # Build directed edges for C (0) - H (1)
    edge_index_list.append([0, 1])
    edge_blocks.append(CH_block)
    edge_dists.append(CH_dist)
    edge_syms.append(CH_sym)

    edge_index_list.append([1, 0])
    edge_blocks.append(CH_block)
    edge_dists.append(CH_dist)
    edge_syms.append(CH_sym)

    # H (1) - O (2)
    edge_index_list.append([1, 2])
    edge_blocks.append(HO_block)
    edge_dists.append(HO_dist)
    edge_syms.append(HO_sym)

    edge_index_list.append([2, 1])
    edge_blocks.append(HO_block)
    edge_dists.append(HO_dist)
    edge_syms.append(HO_sym)

    # C (0) - O (2)
    edge_index_list.append([0, 2])
    edge_blocks.append(CO_block)
    edge_dists.append(CO_dist)
    edge_syms.append(CO_sym)

    edge_index_list.append([2, 0])
    edge_blocks.append(CO_block)
    edge_dists.append(CO_dist)
    edge_syms.append(CO_sym)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_dists_tensor = torch.tensor(edge_dists, dtype=torch.float)

    # Build the Data object
    data = Data(edge_index=edge_index)
    data.atom_sym = atom_syms
    data.center_blocks = center_blocks
    data.edge_blocks = edge_blocks
    data.edge_dist = edge_dists_tensor
    data.edge_pair_sym = edge_syms

    return data

if __name__ == "__main__":
    # Create the example Data
    data = create_example_data()

    # Build a NetworkX DiGraph and add nodes + directed edges
    G = nx.DiGraph()
    for node_idx, sym in enumerate(data.atom_sym):
        G.add_node(node_idx, label=sym)

    for e_idx in range(data.edge_index.shape[1]):
        src, dst = data.edge_index[:, e_idx].tolist()
        G.add_edge(src, dst, label=data.edge_pair_sym[e_idx])

    # Choose fixed positions for visualization
    pos = {0: (0, 0), 1: (1, 1), 2: (1, -1)}

    plt.figure(figsize=(5, 5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={i: data.atom_sym[i] for i in range(len(data.atom_sym))},
        node_size=800,
        node_color="lightblue",
        font_size=12,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
    plt.axis("off")
    plt.title("Example PyG Data (C-H-O) Visualization")
    plt.show()

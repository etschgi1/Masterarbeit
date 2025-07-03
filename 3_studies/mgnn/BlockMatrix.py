import numpy as np
from scf_guess_tools import Backend, load
from pyscf import scf
from itertools import combinations_with_replacement
import os
import matplotlib.pyplot as plt
import pyscf
from copy import deepcopy

example_mol_path = "../../datasets/QM9/xyz_c5h4n2o2/dsgdb9nsd_022700.xyz"

class Block(np.ndarray):
    def __new__(cls, input_array, block_type, atoms, ls, base_mat=None, base_ids=None):
        obj = np.asarray(input_array).view(cls)
        obj.numpy = np.asarray(input_array)
        obj.block_type = block_type
        obj.atoms = atoms
        obj.base_mat = base_mat
        obj.base_ids = base_ids
        obj.ls = ls
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.numpy = getattr(obj, 'numpy', None)
        self.block_type = getattr(obj, 'block_type', None)
        self.atoms = getattr(obj, 'atoms', None)
        self.base_mat = getattr(obj, 'base_mat', None)
        self.base_ids = getattr(obj, 'base_ids', None)
        self.ls = getattr(obj, 'ls', None)
    
    def __repr__(self):
        base_repr = super().__repr__()
        custom_repr = f"Block(shape={self.shape}, block_type={self.block_type}, atoms={self.atoms}, base_ids={self.base_ids}, ls={self.ls})"
        return f"{custom_repr}\n{base_repr}"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        array_inputs = tuple(np.asarray(inp) if isinstance(inp, Block) else inp for inp in inputs)
    
        try:
            res = getattr(ufunc, method)(*array_inputs, **kwargs)
        except TypeError:
            return NotImplemented
        
        blocks = [inp for inp in inputs if isinstance(inp, Block)]
        if len(blocks) == 1 or (len(blocks) == 2 and blocks[0] is blocks[1]):
            res_block_type = blocks[0].block_type
        else: 
            res_block_type = "mixed"
        if isinstance(res, np.ndarray): 
            res = res.view(Block)
            res.block_type = res_block_type
        return res
    def _replace(self, new_values):
        """Change the values of the block"""
        if not isinstance(new_values, np.ndarray):
            raise ValueError("new_values must be a numpy array")
        self[...] = new_values
        self.numpy = new_values.copy()
        return self

class BlockMatrix(): 
    def __init__(self, mol, Matrix=None):
        """mol: pyscf.Mole object
        Currently only creates block matrices for overlap"""
        assert type(mol) in [scf.gto.Mole], "mol must be a pyscf Mole object"
        self.mol = mol
        self.Matrix = self.mol.intor("int1e_ovlp") if Matrix is None else Matrix

        self.blocks = self.generate_all_blocks(self.Matrix) 
    
    
    @classmethod
    def from_blocks(cls, mol, blocks):
        """Create a BlockMatrix from a dictionary of blocks"""
        instance = cls(mol)
        instance.blocks = blocks
        instance.Matrix = np.zeros((mol.nao, mol.nao))
        for block in blocks.values():
            # symmetric assignment
            instance.Matrix[np.ix_(block.base_ids[0], block.base_ids[1])] = block.numpy
            instance.Matrix[np.ix_(block.base_ids[1], block.base_ids[0])] = block.numpy.T
        return instance
    
    def copy(self):
        """Create a copy of the BlockMatrix"""
        new_instance = BlockMatrix(self.mol, self.Matrix.copy())
        new_instance.blocks = self.generate_all_blocks(new_instance.Matrix)
        return new_instance

    def get_overlap(self): 
        """Return the overlap matrix"""
        if self.Matrix is None: 
            return self.Matrix
        else: 
            raise ValueError("Matrix - set get_overlap not supported")
    
    def generate_all_blocks(self, matrix):
        """Generate all (homo and hetero) atom pair blocks"""
        aoslice_per_atom = self.mol.aoslice_by_atom()
        ao_labels = self.mol.ao_labels(fmt=False)


        atom_keys = np.array([f"{x[0]}_{x[1]}" for x in ao_labels])
        atom_ids = np.array([f"{x[0]}_{x[1]}" for x in aoslice_per_atom])  

        unique_atom_keys = self.unique_keep_order(atom_keys)
        unique_atom_ids = self.unique_keep_order(atom_ids)

        indices = [np.arange(start=sl[2], stop=sl[3]) for sl in aoslice_per_atom]

        block_dict = {}
        for i, j in combinations_with_replacement(range(len(indices)), 2):
            key_i = unique_atom_keys[i]
            key_j = unique_atom_keys[j]


            idx_i = indices[i]
            idx_j = indices[j]

            block_key = f"{key_i}-{key_j}"
            block_matrix_view = matrix[idx_i[0]:idx_i[-1]+1, idx_j[0]:idx_j[-1]+1] # create a view to track changes in Blocks

            if key_i == key_j: 
                block_type = "center"
            elif key_i.split("_")[1] == key_j.split("_")[1]:
                block_type = "homo"
            else:
                block_type = "hetero"
            atoms = [key_i.split('_')[0], key_i.split('_')[1], key_j.split('_')[0], key_j.split('_')[1]]

            base_ids = [np.arange(idx_i[0], idx_i[-1]+1) , np.arange(idx_j[0], idx_j[-1]+1)]

            ls = ([ao_labels[i][2] for i in base_ids[0]], [ao_labels[i][2] for i in base_ids[1]])
            block_dict[block_key] = Block(block_matrix_view, base_mat = matrix.view(), base_ids=base_ids, block_type=block_type, atoms=atoms, ls=ls)

        return block_dict
    
    def _plot_blocks(self, masked_array, labels="all", **kwargs): 
        """Plot the blocks of the matrix"""

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 10)))
        imshow_args = kwargs.get("imshow_args", {})
        im = ax.imshow(masked_array, **imshow_args)
        cbar = fig.colorbar(im)
        title_ = kwargs.get("title", "Matrix")
        cbar.set_label(title_.replace("Matrix", ""))
        ax.set_title(kwargs.get("cbar_title", ""))

        # set labels
        labeltext = self.mol.ao_labels(fmt=False)
        if labels.lower() == "atoms":
            from collections import defaultdict
            atom_groups = defaultdict(list)
            for i, label in enumerate(self.mol.ao_labels()):
                key = ' '.join(label.split()[:2])
                atom_groups[key].append(i)

            tick_pos = [np.mean(indices) for indices in atom_groups.values()]
            labeltext = [f"{k.split()[1]}$_{{{k.split()[0]}}}$" for k in atom_groups.keys()]

        elif labels.lower() == "all":
            labeltext = [f"{x[1]}_{x[0]} {x[2]}" for x in labeltext]
            tick_pos = range(len(labeltext))

        else: 
            raise ValueError("Labels must be 'atoms' or 'all'")
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(labeltext, rotation=90, fontsize=7, va='bottom')
        ax.set_yticklabels(labeltext, fontsize=7, ha='left')
        ax.tick_params(axis='x', labelbottom=True, pad=30)
        ax.tick_params(axis='y', labelleft=True, pad=30)
        plt.show()
        return fig, ax

    def plot_blocks_by_element(self, element): 
        pass # TODO implement per element plotting

    def plot_blocks_by_type(self, block_type, mirror_mat = False, **kwargs):
        """Plot all blocks of a certain type"""
        block_type = block_type.lower()
        if block_type == "all": 
            return self._plot_blocks(self.Matrix, **kwargs)
        out = np.full(self.Matrix.shape, np.nan)
        for block in self.blocks.values(): 
            if block.block_type == block_type: 
                out[np.ix_(block.base_ids[0], block.base_ids[1])] = block
        if mirror_mat:
            # out = out + out.T - np.diag(np.diag(out))
            out = np.tril(out.T) + np.triu(out, 1)
        return self._plot_blocks(out, **kwargs)

    def unique_keep_order(self, arr):
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]

    def get_blocks_by_atom(self, atom_str, block_type=None): 
        """Get all blocks for a certain atom and optionally a certain block type"""
        out = []
        for block in self.blocks.values(): 
            if (atom_str is None or atom_str in block.atoms) and (block_type is None or block.block_type == block_type): 
                out.append(block)
        return out
    
    def get_blocks_by_type(self, block_type):
        """Get all blocks of a certain type"""
        return self.get_blocks_by_atom(None, block_type=block_type)
    
    def get_blocks(self):
        """Get all blocks"""
        return self.get_blocks_by_atom(None, block_type=None)
    
    def get_diagonal(self): 
        """Get the diagonal of the matrix"""
        return np.diag(self.Matrix)
    

if __name__ == "__main__": 
    cur_path = os.path.dirname(__file__)
    os.chdir(cur_path)

    mol = load(example_mol_path, Backend.PY, symmetry=False).native
    block_matrix = BlockMatrix(mol)

    print(len(block_matrix.get_blocks_by_atom(None, block_type=None)))
    
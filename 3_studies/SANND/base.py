import numpy as np
from scf_guess_tools import Backend, load
from pyscf import scf
from itertools import combinations_with_replacement
import os
example_mol_path = "../../datasets/QM9/xyz_c5h4n2o2/dsgdb9nsd_022700.xyz"

class Block(np.ndarray):
    def __new__(cls, input_array, block_type, atoms, base_mat=None, base_ids=None):
        obj = np.asarray(input_array).view(cls)
        obj.block_type = block_type
        obj.atoms = atoms
        obj.base_mat = base_mat
        obj.base_ids = base_ids
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.block_type = getattr(obj, 'block_type', None)
        self.atoms = getattr(obj, 'atoms', None)
        self.base_mat = getattr(obj, 'base_mat', None)
        self.base_ids = getattr(obj, 'base_ids', None)
    
    def __repr__(self):
        return f"Block(shape={self.shape}, block_type={self.block_type}, atoms = {self.atoms}, base_ids = {self.base_ids})"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        res = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        blocks = [inp for inp in inputs if isinstance(inp, Block)]
        if len(blocks) == 1 or (len(blocks) == 2 and blocks[0] is blocks[1]):
            res_block_type = blocks[0].block_type
        else: 
            res_block_type = "mixed"
        if isinstance(res, np.ndarray): 
            res = res.view(Block)
            res.block_type = res_block_type
        return res


class BlockMatrix(): 
    def __init__(self, mol):
        """mol: pyscf.Mole object
        Currently only creates block matrices for overlap"""
        self.mol = mol
        self.overlap = self.mol.intor("int1e_ovlp")
        self.blocks = self.generate_all_blocks(self.overlap)
        print(self.blocks)


    def generate_center_blocks(self, matrix):
        """Generate self-overlap (homo) atom blocks"""
        aoslice_per_atom = self.mol.aoslice_by_atom()
        indices = [np.arange(start=x[2], stop=x[3]) for x in aoslice_per_atom]
        print(indices)
        ao_labels = self.mol.ao_labels(fmt=False)
        atoms =  self.unique_keep_order(np.array([f"{x[0]}_{x[1]}-{x[0]}_{x[1]}" for x in ao_labels]))
        atomblocks = {atomkey: matrix[np.ix_(index, index)] for atomkey, index in zip(atoms, indices)}
        return atomblocks
    
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
            block_matrix = matrix[np.ix_(idx_i, idx_j)]

            if key_i == key_j: 
                block_type = "center"
            elif key_i.split("_")[1] == key_j.split("_")[1]:
                block_type = "homo"
            else:
                block_type = "hetero"
            atoms = [key_i.split('_')[0], key_i.split('_')[1], key_j.split('_')[0], key_j.split('_')[1]]
            int_l = lambda x: list(map(int, x.split("_")))
            base_ids = [np.arange(*int_l(unique_atom_ids[i])) , np.arange(*int_l(unique_atom_ids[j]))]
            block_dict[block_key] = Block(block_matrix, base_mat = matrix.view(), base_ids=base_ids, block_type=block_type, atoms=atoms)

        return block_dict
    


    def unique_keep_order(self, arr):
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]

    def info(self): 
        print(self.mol)


if __name__ == "__main__": 
    cur_path = os.path.dirname(__file__)
    os.chdir(cur_path)

    mol = load(example_mol_path, Backend.PY, symmetry=False).native
    block_matrix = BlockMatrix(mol)
    block_matrix.info()
    
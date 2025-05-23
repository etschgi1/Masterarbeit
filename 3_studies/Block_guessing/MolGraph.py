import os
from rdkit.Chem import rdmolfiles
from scf_guess_tools import Backend, load
from BlockMatrix import BlockMatrix, Block

BASIS_PATH = "../../scripts/6-31g_2df_p_custom_nwchem.gbs"


class MolGraph(): 
    """Class to represent a molecular graph."""

    def __init__(self, Molecule):

        self.mol = Molecule
        self.atoms = self.mol._atom
        self.atom_keys = [a[0] for a in self.atoms]
        self.atom_coords = [a[1] for a in self.atoms]
        self.block_ovlp = BlockMatrix(self.mol)
        # self.atoms = [self.mol.atom_]
    

if __name__ == "__main__": 
    cur_path = os.path.dirname(__file__)
    os.chdir(cur_path)

    mol = load("data/h2o_1.xyz", Backend.PY, basis = BASIS_PATH, symmetry=False).native
    g = MolGraph(mol)
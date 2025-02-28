import os
from openbabel import pybel

# Defs:
organic_constituents = [6, 7, 8, 9, 15, 16, 17, 35, 53]

# SMARTS patterns for functional groups
functional_groups = {
    "Alcohol": pybel.Smarts("[#6][OX2H]"),
    "Carboxylic Acid": pybel.Smarts("[CX3](=O)[OX2H1]"),
    "Aldehyde": pybel.Smarts("[CX3H1](=O)[#6]"),
    "Ketone": pybel.Smarts("[CX3](=O)[#6]"),
    "Ester": pybel.Smarts("[CX3](=O)[OX2][#6]"),
    "Ether": pybel.Smarts("[OD2]([#6])[#6]"),
    "Amine": pybel.Smarts("[NX3;H2,H1,H0;!$(NC=O)]"),
    "Amide": pybel.Smarts("[NX3][CX3](=O)[#6]"),
    "Nitrile": pybel.Smarts("[CX2]#N"),
    "Alkene": pybel.Smarts("C=C"),
    "Alkyne": pybel.Smarts("C#C"),
}


def get_xyz_file_paths(folder):
    xyz_file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.xyz'):
                xyz_file_paths.append(os.path.join(root, file))
    return xyz_file_paths

def categorize_organic(molecule): 
    mol = pybel.readstring("inchi", molecule)
    atoms = [atom.atomicnum for atom in mol.atoms]
    if all([atom in organic_constituents for atom in atoms]): 
        return "organic"
    else: 
        return "inorganic"
    

def categorize_functional_group(smiles):
    mol = pybel.readstring("smi", smiles)
    
    detected_groups = [name for name, pattern in functional_groups.items() if pattern.findall(mol)]
    
    return ", ".join(detected_groups) if detected_groups else "Unknown"
    
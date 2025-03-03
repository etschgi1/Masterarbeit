import os
from openbabel import pybel

# Defs:
organic_constituents = [6, 7, 8, 9, 15, 16, 17, 35, 53]
metals = {
            3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,39,40,41,42,43,44,
            45,46,47,48,49,55,56,57,58,59,60,61,62,63,64,65,66,72,73,74,75,76,77,78,79,
            80,81,82,83,87,88,89,90,91,92,93,94,95,96
        }

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
    "Nitro": pybel.Smarts("[N+](=O)[O-]"),
    "Thiol": pybel.Smarts("[SX2H]"),
    "Halogenalkane": pybel.Smarts("[CX4][F,Cl,Br,I]"),
    # Cyclic functional groups
    "Furan": pybel.Smarts("c1ccco1"),
    "Thiophene": pybel.Smarts("c1ccsc1"),
    "Pyrrole": pybel.Smarts("c1cc[nH]c1"),
    "Epoxide": pybel.Smarts("C1OC1")
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
    elif any([atom in metals for atom in atoms]) and any([atom == 6 for atom in atoms]):
        return "organometallic"
    else: 
        return "inorganic"
    

def categorize_functional_group(smiles):
    mol = pybel.readstring("smi", smiles)
    print(mol)
    detected_groups = [name for name, pattern in functional_groups.items() if pattern.findall(mol)]
    if detected_groups:
        return ", ".join(detected_groups)
    else:
        has_metal = any(a in metals for a in [atom.atomicnum for atom in mol.atoms])
        carbon_inside = 6 in [atom.atomicnum for atom in mol.atoms]
        if has_metal and carbon_inside:
            return "(Fallback) Organometallic"
        elif has_metal:
            return "(Fallback) Inorganic"
        elif carbon_inside:
            return "(Fallback) Organic"
        else:
            return "(Fallback) Inorganic"

def get_atom_count(smiles): 
    mol = pybel.readstring("smi", smiles)
    return len(mol.atoms)

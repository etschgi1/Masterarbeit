from rdkit import Chem
import os, re
import numpy as np

# FOLDER = "/home/etschgi1/REPOS/Masterarbeit/datasets/QM9/xyz_c7h10o2"
# FOLDER = "/home/etschgi1/REPOS/Masterarbeit/datasets/QM9/xyz_c5h4n2o2"
# FOLDER = "/home/ewachmann/REPOS/Masterarbeit/scf_guess_datasets/scf_guess_datasets/qm9_isomeres/xyz"
FOLDER = "/home/ewachmann/REPOS/Masterarbeit/scf_guess_datasets/scf_guess_datasets/qm9/xyz"

VALENCE_ELECTRONS = {
    1: 1,   # H
    6: 4,   # C
    7: 5,   # N
    8: 6,   # O
    9: 7,   # F
    16: 6,  # S
}

def get_multiplicity_smiles(smiles): 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        raise ValueError("Invalid Smiles string")
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    total_valence_electrons = sum(
        VALENCE_ELECTRONS.get(atom.GetAtomicNum(), 0) for atom in mol.GetAtoms()
    )
    multiplicity = 1 if total_valence_electrons % 2 == 0 else 2

    return charge, multiplicity


charge_list, mult_list = [], []
charge, mult = 0, 1
overwrite_flag = False
def add_charge_and_mult(folder_path): 
    global overwrite_flag, charge, mult
    for file in os.listdir(folder_path): 
        if not file.endswith(".xyz"):
            continue
        filepath = os.path.join(folder_path, file)
        with open(filepath, "r") as f: 
            content = f.read()
            # check if file is already preprocessed
            if "charge" in content:
                print(f"File {filepath} already preprocessed") 
                continue
            content = content.split("\n")
            if not overwrite_flag:
                inchi_ = content[-2].split("\t")[0].split("=")[-1]
                smiles = content[-3].split("\t")[0]
                try:
                    charge, mult = get_multiplicity_smiles(smiles)
                except Exception as e: 
                    print(e)
                    in_ = input("Fallback to charge 0 multiplicity 1?")
                    if in_.lower() == "y":
                        overwrite_flag = True
            content[1] = content[1] + f"charge {charge} multiplicity {mult}"
        with open(filepath, "w+") as f:
            f.write("\n".join(content))


if __name__ == "__main__": 
    add_charge_and_mult(FOLDER)
    nr_files = len([f for f in os.listdir(FOLDER) if f.endswith(".xyz")])
    print(f"Added charge and multiplicity to {nr_files} files in {FOLDER}")
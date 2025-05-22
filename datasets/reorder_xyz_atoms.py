import os
import numpy as np
from mendeleev import element

XYZ_INPUT_FOLDER = "/home/etschgi1/REPOS/Masterarbeit/datasets/QM9/xyz_c7h10o2"
OUTPUT_FOLDER = "/home/etschgi1/REPOS/Masterarbeit/datasets/QM9/xyz_c7h10o2_sorted"
if not os.path.exists(OUTPUT_FOLDER): 
    os.makedirs(OUTPUT_FOLDER)

elem_ord = {}

def sort_xyz_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0])
    atom_lines = lines[2:2 + num_atoms]
    rest = lines[2 + num_atoms:]

    # Sort the atom lines based on the first column (atom type)
    try: 
        sorted_atom_lines = sorted(atom_lines, key=lambda x: elem_ord[x.split()[0]])[::-1]
    except KeyError as e:
        elems = [x.split()[0] for x in atom_lines]
        for x in elems:
            if x not in elem_ord:
                try:
                    elem_ord[x.split()[0]] = element(x).atomic_number
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Element {x} not found in mendeleev.")
                    continue
    finally:
        sorted_atom_lines = sorted(atom_lines, key=lambda x: elem_ord[x.split()[0]])[::-1]
    # Write the sorted lines to the output file
    with open(output_file, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(lines[1])  # Write the comment line
        f.writelines(sorted_atom_lines)
        f.writelines(rest)

if __name__ == "__main__":
    for file in os.listdir(XYZ_INPUT_FOLDER):
        file_path = os.path.join(XYZ_INPUT_FOLDER, file)
        if not file.endswith(".xyz"):
            continue
        dest_path = os.path.join(OUTPUT_FOLDER, file)
        sort_xyz_file(file_path, dest_path)


import os, shutil
import numpy as np

XYZ_INPUT_FOLDER = "datasets/QM9/xyz"
OUTPUT_FOLDER = "/home/etschgi1/REPOS/Masterarbeit/datasets/QM9/xyz_c7h10o2"

def filterc7h10o2(raw_text): 
    nr_atoms = int(raw_text[0])
    if nr_atoms != 19: 
        return False 
    start_of_lines = [l.split("\t")[0] for l in raw_text]
    start_of_lines = np.array([x for x in start_of_lines if x in ["C", "H", "O"]])
    _, counts = np.unique(start_of_lines, return_counts=True)
    if np.array_equal(counts, [7,10,2]):
        return True
    return False



def filter_ds(input_folder, output_folder, filter, copy=True): 
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if not file.endswith(".xyz"):
            continue
        with open(file_path, "r") as f:
            if(filter(f.readlines())):
                dest_path = os.path.join(output_folder, file)
                # check if file is already in output folder
                if os.path.exists(dest_path):
                    print(f"File {file} already in output folder, skipping...")
                    continue
                if copy:
                    shutil.copy(file_path, dest_path)
                else:
                    shutil.move(file_path, dest_path)


filter_ds(XYZ_INPUT_FOLDER, OUTPUT_FOLDER, filterc7h10o2)
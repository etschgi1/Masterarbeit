import numpy as np
import os, re, multiprocessing
import h5py

#! Strange workaround idk why it didn't work before
slurm_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
os.environ["OMP_NUM_THREADS"] = str(slurm_threads)
os.environ["MKL_NUM_THREADS"] = str(slurm_threads)
os.environ["NUMEXPR_MAX_THREADS"] = str(slurm_threads)  # NumPy multi-threading

from scf_guess_tools import Backend, load, calculate
import psi4, pyscf
from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize
pyscf.lib.num_threads(slurm_threads)
psi4.set_num_threads(slurm_threads)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

XYZ_INPUT_FOLDER = "../../../datasets/QM9/xyz_c7h10o2"

def setup_test(samples=10, seed=42):
    import random
    global XYZ_INPUT_FOLDER
    cur_path = os.path.dirname(os.path.abspath(__file__))
    test_input_folder = os.path.join(cur_path, "test_data")
    if not os.path.exists(test_input_folder):
        os.makedirs(test_input_folder)
    if os.listdir(test_input_folder):
        print(f"Warning: {test_input_folder} is not empty. Skipping sampling.")
        XYZ_INPUT_FOLDER = test_input_folder
        return os.listdir(test_input_folder)
            
    files = os.listdir(XYZ_INPUT_FOLDER)
    random.seed(seed)
    random_sample = random.sample(files, samples)
    for file in random_sample:
        os.system(f"cp {os.path.join(XYZ_INPUT_FOLDER, file)} {os.path.join(test_input_folder, file)}")
    XYZ_INPUT_FOLDER = test_input_folder
    print(f"Setup test with {samples} samples in {XYZ_INPUT_FOLDER}")
    return os.listdir(XYZ_INPUT_FOLDER)

def main(): 
    global XYZ_INPUT_FOLDER
    import time
    filenames = setup_test(samples=5)
    optimized_out_path = os.path.join(XYZ_INPUT_FOLDER, "../optimized")
    if not os.path.exists(optimized_out_path):
        os.makedirs(optimized_out_path)
    print(f"Got: {filenames}")
    start_time = time.time()
    skip_flag = False
    for filename in filenames:
        print(f"Starting with {filename}")
        if skip_flag == False and filename in os.listdir(optimized_out_path):
            print(f"Already optimized {filename}. Recalc? (y/N/ya/na).")
            inp_ = input().lower()
            if inp_ == "ya":
                print("Deleting all optimized files")
                for file in os.listdir(optimized_out_path):
                    os.remove(os.path.join(optimized_out_path, file))
            elif inp_ == "y":
                print(f"Deleting {filename}")
                os.remove(os.path.join(optimized_out_path, filename))
            elif inp_ == "na":
                skip_flag = True
                continue
            else:
                print("Skipping...")
                continue
            
        mol = gto.Mole()
        mol.fromfile(os.path.join(XYZ_INPUT_FOLDER, filename))
        with open(os.path.join(XYZ_INPUT_FOLDER, filename), "r") as f:
            lines = f.readlines()
            first_two_lines = lines[:2]
        mol.basis = 'aug-cc-pVDZ'
        mol.build()

        mf = scf.RKS(mol)
        mf.xc = 'WB97X-V'

        mol_eq = optimize(mf, maxsteps=100)
        print(f"Done with {filename}")
        output_file = os.path.join(optimized_out_path, filename)
        with open(output_file, "w") as f:
            #write top 2 lines 
            f.write("".join(first_two_lines))
            f.write(mol_eq.tostring())
        print(f"Optimized geometry saved to {output_file}")
    print(f"Took {time.time() - start_time} seconds to optimize {len(filenames)} files")
    print("All done!")

if __name__ == "__main__":
    main()
    from rms_bench import get_rms
    print(f"RMSD: {get_rms(XYZ_INPUT_FOLDER, os.path.join(XYZ_INPUT_FOLDER, '../optimized'))}")
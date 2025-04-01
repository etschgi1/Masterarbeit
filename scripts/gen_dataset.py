import numpy as np
import os, re, multiprocessing
import h5py

#! Strange workaround idk why it didn't work before
slurm_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
os.environ["OMP_NUM_THREADS"] = str(slurm_threads)
os.environ["MKL_NUM_THREADS"] = str(slurm_threads)
os.environ["NUMEXPR_MAX_THREADS"] = str(slurm_threads)  # NumPy multi-threading

from scf_guess_tools import Backend
from to_cache import density_fock_overlap
import psi4, pyscf
pyscf.lib.num_threads(slurm_threads)
psi4.set_num_threads(slurm_threads)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

XYZ_INPUT_FOLDER = "../datasets/QM9/xyz_c7h10o2"
OUTPUT_ROOT = "../datasets/QM9/out"

class GenDataset: 
    def __init__(self, backend, xyz_root, output_folder, calc_basis, options={}):
        self.input_folder = xyz_root
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder): 
            os.makedirs(self.output_folder)
        self.backend = backend
        self.backend_name = "pyscf" if self.backend == Backend.PY else "psi4" if self.backend == Backend.PSI else "unknown"
        self.calc_basis = calc_basis
        self.options = options if options != {} else None
        self.files = None
        self.files = self.get_input_files()
        for k,v in options.items():
            setattr(self, k.lower(), v)
        if hasattr(self, "output_folder_name"): 
            self.output_folder = os.path.join(OUTPUT_ROOT, self.output_folder_name)

        # if hasattr(self, "nr_threads"): 
        #     print(f"Set number of threads to {slurm_threads}")
        #     os.environ["OMP_NUM_THREADS"] = str(slurm_threads)
        #     os.environ["MKL_NUM_THREADS"] = str(slurm_threads)

        #     if self.backend == Backend.PSI: 
        #         psi4.set_num_threads(slurm_threads)
        # else: # set all threads we can get
        #     os.environ["OMP_NUM_THREADS"] = str(slurm_threads)
        #     os.environ["MKL_NUM_THREADS"] = str(slurm_threads)
    def get_input_files(self): 
        if self.files: 
            return self.files
        files = []
        for file in os.listdir(self.input_folder): 
            if file.endswith(".xyz"): 
                files.append(os.path.join(self.input_folder, file))
        print(f"Got {len(files)} files from {self.input_folder}!")
        return files
    
    def get_ref_energy(self, file):
        with open(file, "r") as f: 
            l2 = f.readlines()[1]
        E0 = float(l2.split()[12])
        ZPE = float(l2.split()[11])
        return E0 - ZPE
    
    def create_valid_xyz(self, file): 
        """Create a valid xyz which is suitable for psi / pyscf"""
        with open(file, "r") as f: 
            raw = f.readlines()
        filename = os.path.basename(file).split(".")[0]
        cleaned = raw[:2]
        for c, l in enumerate(raw[2:]): #first 2 lines ok!
            atom_sym = l.split("\t")[0]
            if not re.match("^[A-Z][a-z]?$", atom_sym): 
                break
            xyz_data = "\t".join(l.split("\t")[:4])+"\n"
            cleaned.append(xyz_data)
        temp_path = f"/tmp/{filename}_valid.xyz" # got enough for now
        with open(temp_path, "w") as f: 
            f.writelines(cleaned)
        return temp_path

    def gen_from_files(self, guess_type): 
        cache_folder = os.path.join(self.output_folder, self.backend_name)
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        print(f"Using following for density_fock_overlap:")
        print(f"File: e.g. {self.files[0]} (total: {len(self.files)})")
        print(f"Filename: **")
        print(f"Method: {self.method}")
        print(f"Calculation Basis: {self.calc_basis}")
        print(f"Functional: {self.functional}")
        print(f"Guess Type: {guess_type}")
        print(f"Backend: pyscf")
        print(f"Cache Folder: {cache_folder}")
        for c, file in enumerate(self.files): 
            try:
                if "early_stop" in self.options.keys() and self.options["early_stop"] <= c: 
                    break
                filename = os.path.basename(file).strip()
                print(f"\n---\nProcessing: {c+1}/{len(self.files)}: {filename}")
                file = self.create_valid_xyz(file) #! only needed for psi4 at the moment 
                ret = density_fock_overlap(file, filename, self.method, self.calc_basis, self.functional, guess=guess_type, backend="pyscf", cache=cache_folder)
                if any([x is None for x in ret]): 
                    print("Not all data available!")
                else: 
                    print(f"Got all data for {filename}")

            except Exception as e: #some files seem to fail
                print("Failed :(")
                print(e)
            # finally: 
            #     try:
            #         scf_energy = wf.electronic_energy() + wf.nuclear_repulsion_energy()
            #         print(f"Diff to reference energy: {refernce_energy - scf_energy}")
            #         print(f"SCF Energy: {scf_energy}")
            #         print(f"Reference Energy: {refernce_energy}")
            #     except: 
            #         print("No diff")
        
    def gen(self): 
        if self.backend.value == "Psi": #Psi4
            assert "psi4_guess_type" in dir(self), "Missing guess type for psi4 engine"
            if "PSI_SCRATCH" not in os.environ: 
                psi_scratch_path = os.path.abspath(os.path.join(self.input_folder, ".."))
                os.environ["PSI_SCRATCH"] = psi_scratch_path
                print(f"PSI_SCRATCH is not set. Using: {psi_scratch_path}")
            else:
                print(f"PSI_SCRATCH is already set to: {os.environ['PSI_SCRATCH']}")
            print("Start psi4")

            self.gen_from_files(self.options["psi4_guess_type"])

        elif self.backend.value == "Py": 
            assert "pyscf_guess_type" in dir(self), "Missing guess type for pyscf engine"
            print("Start pyscf")
            self.gen_from_files(self.options["pyscf_guess_type"])

        else: 
            raise Exception(f"Backend {self.backend.value} not supported!")
        





if __name__ == "__main__": 
    import time
    gds_options = {"pyscf_guess_type": "minao", "output_folder_name": "c7h10o2_b3lypg_6-31G(2df,p)", "nr_threads": 32, "method":"dft", "functional":"b3lypg", "early_stop": 2} 
    basis_path = "6-31g_2df_p_custom_nwchem.gbs"
    gds = GenDataset(Backend.PY, XYZ_INPUT_FOLDER, OUTPUT_ROOT, basis_path, gds_options)
    start_ = time.time()
    gds.gen()
    print(f"Time elapsed (pyscf): {time.time() - start_}")
    
    # ! way slower than PY?! - 

    # start_ = time.time()
    # gds_options["functional"] = "wb97x-v"
    # gds = GenDataset(Backend.PSI, XYZ_INPUT_FOLDER, OUTPUT_ROOT, "aug-cc-pVDZ", gds_options)
    # gds.gen()
    # print(f"Time elapsed (psi4): {time.time() - start_}")
    
    
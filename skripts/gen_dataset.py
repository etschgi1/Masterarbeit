import numpy as np
from scf_guess_tools import PyEngine, PsiEngine
import os

XYZ_INPUT_FOLDER = "/home/ewachmann/REPOS/Masterarbeit/datasets/QM9/xyz"
OUTPUT_ROOT = "/home/ewachmann/REPOS/Masterarbeit/datasets/QM9/out"

class GenDataset: 
    def __init__(self, xyz_root, output_folder, calc_basis, options={}):
        self.input_folder = xyz_root
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder): 
            os.makedirs(self.output_folder)
        self.calc_basis = calc_basis
        self.options = options if options != {} else None
        self.files = None
        self.files = self.get_input_files()
        for k,v in options.items():
            setattr(self, k.lower(), v)

    def get_input_files(self): 
        if self.files: 
            return self.files
        files = []
        for file in os.listdir(self.input_folder): 
            if file.endswith(".xyz"): 
                files.append(os.path.join(self.input_folder, file))
        print(f"Got {len(files)} files from {self.input_folder}!")
        return files
    
    def genfromfiles(self, engine, guess_type): 
        engine_name = engine.__repr__()
        output_folder = os.path.join(self.output_folder, engine_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for c, file in enumerate(self.files): 
            if "early_stop" in self.options.keys() and self.options["early_stop"] < c: 
                break
            mol = engine.load(file) #! TODO xyz not in our format!
            wf = engine.calculate(mol, self.calc_basis, guess_type)
            D = wf.D
            #! TODO save these things


    
    def gen(self): 
        assert any([x in ["psi4", "pyscf"] for x in dir(self)]), "Specify engine using psi4 or pyscf"
        if self.psi4: 
            assert "psi4_guess_type" in dir(self), "Missing guess type for psi4 engine"
            if "PSI_SCRATCH" not in os.environ: 
                psi_scratch_path = os.path.abspath(os.path.join(self.input_folder, ".."))
                os.environ["PSI_SCRATCH"] = psi_scratch_path
                print(f"PSI_SCRATCH nicht gesetzt. Setze auf: {psi_scratch_path}")
            else:
                print(f"PSI_SCRATCH ist bereits gesetzt auf: {os.environ['PSI_SCRATCH']}")

            print("Start psi4")
            self.genfromfiles(self.psi4(), self.options["psi4_guess_type"])

        if self.pyscf: 
            assert "pyscf_guess_type" in dir(self), "Missing guess type for pyscf engine"
            print("Start pyscf")
            self.genfromfiles(self.pyscf(), self.options["pyscf_guess_type"])





if __name__ == "__main__": 
    gds_options = {"psi4": PsiEngine, "pyscf": PyEngine, "psi4_guess_type": "core", "pyscf_guess_type": "1e", "early_stop": 10} #! Symmetry???
    gds = GenDataset(XYZ_INPUT_FOLDER, OUTPUT_ROOT, "pcseg-1", gds_options)
    gds.gen()
    
    
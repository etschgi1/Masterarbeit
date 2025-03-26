import time, random, os, re
from scf_guess_tools import Backend, load, calculate
rng_seed = 42
random.seed(rng_seed)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_ref_energy(file):
    with open(file, "r") as f: 
        l2 = f.readlines()[1]
    return float(l2.split("\t")[11])

def create_valid_xyz(file): # why psi4???
        """Create a valid xyz which is suitable for psi"""
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


xyz_source = "../../datasets/QM9/xyz_c7h10o2/"
all_files = [os.path.join(xyz_source, f) for f in os.listdir(xyz_source) if f.endswith(".xyz")]

# take a small sample (10 for now)
sample_size = 10
files_to_calc = random.sample(all_files, sample_size)

basis_pyscf = "6-31G(2df,p)"
basis_path_psi4 = "../../datasets/basis/6-31g_2df_p.gbs"

pyscf_energies, psi4_energies = [], []
pyscf_times, psi4_times = [], []
reference_energies = [get_ref_energy(f) for f in files_to_calc]

# psi4
backend = Backend.PSI
print("Starting calculations with Psi4")
for i, f in enumerate(files_to_calc):
    print(f"Calculating {i+1}/{sample_size}")
    start_ = time.time()
    valid_file = create_valid_xyz(f)
    mol = load(valid_file, backend)
    wf = calculate(mol, basis_path_psi4, None, method="dft", functional="b3lyp", cache=False) # no cache to get true comp. time
    energy = wf.electronic_energy() + wf.nuclear_repulsion_energy()
    psi4_times.append(time.time() - start_)
    psi4_energies.append(energy)

# pyscf
backend = Backend.PY
print("Starting calculations with PySCF")
for i, f in enumerate(files_to_calc):
    print(f"Calculating {i+1}/{sample_size}")
    start_ = time.time()
    mol = load(f, backend)
    wf = calculate(mol, basis_pyscf, None, method="dft", functional="b3lypg", cache=False) # no cache to get true comp. time
    energy = wf.electronic_energy() + wf.nuclear_repulsion_energy()
    pyscf_times.append(time.time() - start_)
    pyscf_energies.append(energy)


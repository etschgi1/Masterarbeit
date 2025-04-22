import sys, os
from glob import glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))
scripts_path = "../../scripts"
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
from to_cache import density_fock_overlap


train_test_seed = 42
source_path = '../../datasets/QM9/xyz_c7h10o2/'
all_file_paths = glob(os.path.join(source_path, '*.xyz'))
len(all_file_paths)

def load_cached(file_paths, cache_path, basis, guess="minao", method="dft", functional="b3lypg", backend="pyscf"):
    error_list = []
    error_files = []
    focks = []
    used_files = []
    reference_densities = []
    for file in file_paths:
        mol_name = os.path.basename(file).strip()
        print(mol_name)
        try: 
            ret = density_fock_overlap(filepath = file,
                                filename = mol_name,
                                method = method,
                                basis = None,
                                functional = functional,
                                guess = guess,
                                backend = backend,
                                cache = cache_path)
            print(f"Using: file={file} - mol_name={mol_name} - basis={None} - guess={guess} - method={method} - functional={functional}")
            break
        except Exception as e: 
            error_list.append(e)
            error_files.append(mol_name)
            print(f"File {mol_name} error - skipping")
            continue
        if any([r == None for r in ret]): 
            print(f"File {mol_name} bad - skipping")
            continue
        focks.append(ret[1].numpy)
        used_files.append(file)
        reference_densities.append(ret[0].numpy)
    print(f"Got data for: {len(focks)} - bad / no ret: {len(file_paths) - len(focks) - len(error_list)} - errors: {len(error_list)}")
    print(error_files[:5])
    return focks, reference_densities, used_files

ret = load_cached(all_file_paths, "../../datasets/QM9/out/c7h10o2_b3lypg_6-31G(2df,p)/pyscf", basis="6-31g_2df_p_custom_nwchem.gbs")
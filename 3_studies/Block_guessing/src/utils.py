VERBOSE_LEVEL = 3

def dprint(printlevel, *args, **kwargs):
    """Customized printing levels"""
    global VERBOSE_LEVEL
    if printlevel <= VERBOSE_LEVEL:
        print(*args, **kwargs)

def set_verbose(level: int):
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = level


from scf_guess_tools import Backend, cache, load, calculate
import os



@cache(ignore=["filepath", "basis"])
def density_fock_overlap(filepath, 
                         filename, 
                         method = "dft",
                         basis = "6-31G(2df,p)", 
                         functional = "b3lypg", 
                         guess = "minao",
                         backend="pyscf", 
                         ): 
    if backend == "pyscf":
        backend = Backend.PY
    else: 
        raise NotImplementedError(f"Backend {backend} not implemented")
    
    assert filepath.endswith(".xyz"), "File is not an xyz file"
    try: 
        mol = load(filepath, backend=backend)
        print(f"Loaded mol from {filepath}")
    except:
        print(f"Failed to load {filepath}")
        return None, None, None
    try:
        wf = calculate(mol, basis, guess, method=method, functional=functional, cache=False)
    except Exception as e:
        print(f"Failed to calculate {filepath}")
        print(e)
        return None, None, None
    
    density, fock, overlap = None, None, None
    try: 
        density = wf.density()
    except: 
        print("No density matrix available")
    try: 
        fock = wf.fock()
    except: 
        print("No fock matrix available")
    try: 
        overlap = wf.overlap()
    except: 
        print("No overlap matrix available")
    try:
        core_hamiltonian = wf.core_hamiltonian()
    except: 
        print("No core hamiltonian available")
    try:
        electronic_energy = wf.electronic_energy()
    except:
        print("No electronic energy available")

    return density, fock, overlap, core_hamiltonian, electronic_energy

#! spoof module name to read from right cache! 
orig = density_fock_overlap.__wrapped__

# force it to look like it came from to_cache
orig.__module__ = "to_cache"

# now re-decorate
density_fock_overlap = cache(ignore=["filepath", "basis"])(orig)

def check_import():
    print("Import worked")
    return True
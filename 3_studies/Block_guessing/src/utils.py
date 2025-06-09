import numpy as np
from scf_guess_tools import Backend, cache, load, calculate


VERBOSE_LEVEL = 3

def dprint(printlevel, *args, **kwargs):
    """Customized printing levels"""
    global VERBOSE_LEVEL
    if printlevel <= VERBOSE_LEVEL:
        print(*args, **kwargs)

def set_verbose(level: int):
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = level





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

def unflatten_triang(flat, N):
    M = np.zeros((N, N))
    iu = np.triu_indices(N)
    M[iu] = flat
    M[(iu[1], iu[0])] = flat 
    return M


def plot_mat_comp(reference, prediction, reshape=False, title="Fock Matrix Comparison", ref_title="Reference", pred_title="Prediction", vmax=1.5, labels1=None, labels2=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import root_mean_squared_error
    diff = reference - prediction
    rmse = root_mean_squared_error(reference, prediction)
    
    reference = unflatten_triang(reference, reshape) if reshape else reference
    prediction = unflatten_triang(prediction, reshape) if reshape else prediction
    diff = unflatten_triang(diff, reshape) if reshape else diff
    
    fig, ax = plt.subplots(1, 4, figsize=(15, 5), width_ratios=[1, 1, 1, 0.1])
    fig.suptitle(f"{title}  |  RMSE: {rmse:.8f}")
    
    ax[0].imshow(reference, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax[0].set_title(ref_title)
    
    ax[1].imshow(prediction, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax[1].set_title(pred_title)
    
    diff_plot = ax[2].imshow(diff, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax[2].set_title("Difference")
    
    if labels1: 
        ax[0].set_xticks(range(len(labels1)))
        ax[0].set_xticklabels(labels1, rotation=90, fontsize=7, va='bottom')
        ax[0].set_yticks(range(len(labels1)))
        ax[0].set_yticklabels(labels1, fontsize=7, ha='left')
        ax[0].tick_params(axis='x', labelbottom=True, pad=30)
        ax[0].tick_params(axis='y', labelleft=True, pad=30)
    if labels2: 
        ax[1].set_xticks(range(len(labels2)))
        ax[1].set_xticklabels(labels2, rotation=90, fontsize=7, va='bottom')
        ax[1].set_yticks(range(len(labels2)))
        ax[1].set_yticklabels(labels2, fontsize=7, ha='left')
        ax[1].tick_params(axis='x', labelbottom=True, pad=30)
        ax[1].tick_params(axis='y', labelleft=True, pad=30)
    
    cbar = fig.colorbar(diff_plot, cax=ax[3])
    cbar.set_label("Difference Scale")
    
    plt.tight_layout()
    plt.show()

from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scf_guess_tools import calculate, load, Backend
from pyscf import scf
from scipy.linalg import eigh
import re
from time import time

def check_positive_definite(S, tol=1e-10):
    eigvals = np.linalg.eigvalsh(S)
    is_pd = np.all(eigvals > tol)
    return is_pd

def density_pred(model, overlap, nocc): 
    fock = model.predict(flatten_triang(overlap).reshape(1, -1))
    return density_from_fock(fock, overlap, nocc)

def density_from_fock(fock, overlap, nocc):
    assert check_positive_definite(overlap)
    _, C = eigh(fock, overlap)
    C_occ = C[:, :nocc]
    density = 2 * C_occ @ C_occ.T 
    return density

def comp_trace(overlap, *args): 
    for arg in args: 
        trace_ = np.trace(arg @ overlap)
        print(f"Trace: {trace_}")

def flatten_triang(M): 
    return M[np.triu_indices(M.shape[0], k=0)]

def unflatten_triang(flat, N):
    M = np.zeros((N, N))
    iu = np.triu_indices(N)
    M[iu] = flat
    M[(iu[1], iu[0])] = flat 
    return M

def flatten_triang_batch(M_batch):
    """Input: (n_samples, N, N) → Output: (n_samples, N*(N+1)//2)"""
    N = M_batch.shape[1]
    iu = np.triu_indices(N)
    return M_batch[:, iu[0], iu[1]]

def unflatten_triang_batch(flat_batch, N):
    """Input: (n_samples, N*(N+1)//2) → Output: (n_samples, N, N)"""
    M_batch = np.zeros((flat_batch.shape[0], N, N))
    iu = np.triu_indices(N)
    M_batch[:, iu[0], iu[1]] = flat_batch
    M_batch[:, iu[1], iu[0]] = flat_batch  
    return M_batch


def perform_calculation(file, density_guess, basis_set, method, functional=None): 
    """Only RHF and RKS are supported."""
    mol = load(file, Backend.PY)
    mol_native = mol.native
    mol_native.basis = basis_set
    mol_native.build()

    if method.lower() == "dft":
        if functional is None:
            raise ValueError("Functional must be specified for DFT calculations.")
        wf = scf.RKS(mol_native)
        wf.xc = functional
    elif method.lower() == "hf":
        wf = scf.RHF(mol_native)
    else:
        raise ValueError("Method must be either 'HF' or 'DFT'.")
    if type(density_guess) == np.ndarray:
        wf.kernel(dm0=density_guess)
    elif type(density_guess) == str:
        assert density_guess in ["atom", "1e", "minao", "huckel", "vsap"], "Density guess must be 'atom', '1e', 'minao', 'huckel' or 'vsap'."
        dm0 = wf.get_init_guess(key=density_guess)
        wf.kernel(dm0=dm0)
    else:
        raise ValueError("Density guess must be either a numpy array or a string.")
    return {"cycles": wf.cycles, "conv": wf.converged, "summary": wf.scf_summary, "wf": wf, "mol": mol_native}

def benchmark_cycles(files, density_guesses, scheme_names, basis_set, method, functional=None, max_samples=None):
    """Benchmarks the number of cycles needed to converge the SCF calculation."""
    results = {}
    assert len(scheme_names) == len(density_guesses)
    for i, scheme in enumerate(scheme_names):
        cur_samples = 0
        results[scheme] = {"cycles": [], "converged": [], "summary": [], "wf": [], "mol": []}
        print("Starting scheme:", scheme)
        if density_guesses[i] is None: # mock array
            density_guesses[i] = [None] * len(files)
        for file, density_guess in zip(files, density_guesses[i]):
            density_guess = density_guess if density_guess is not None else scheme_names[i].lower()
            try:
                result = perform_calculation(file, density_guess, basis_set, method, functional)
                results[scheme]["cycles"].append(result["cycles"])
                results[scheme]["converged"].append(result["conv"])
                results[scheme]["summary"].append(result["summary"])
                results[scheme]["wf"].append(result["wf"])
                results[scheme]["mol"].append(result["mol"])
                print(f"Finished scheme {scheme} for file {file}: {result['cycles']} cycles, converged: {result['conv']}")
                cur_samples += 1
                if max_samples is not None and cur_samples >= max_samples:
                    print(f"Reached max_samples for scheme {scheme}: {max_samples}")
                    break
            except Exception as e:
                print(f"Error processing {file} with scheme {scheme}: {e}")
                results[scheme]["cycles"].append(None)
                results[scheme]["converged"].append(False)
                results[scheme]["summary"].append(None)
                results[scheme]["wf"].append(None)
                results[scheme]["mol"].append(None)
                print(f"Failed scheme {scheme} for file {file}: {e}")
    return results

def plot_mat_comp(reference, prediction, reshape=False, title="Fock Matrix Comparison", ref_title="Reference", pred_title="Prediction", vmax=1.5, labels1=None, labels2=None):
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

def permute_xyz_file(filename, perm_index=None,  tmp_file = "/tmp/perm.xyz"): 
    """Permute the coordinates of a XYZ file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0])
    atom_lines = lines[2:2 + num_atoms]
    perm_index = np.random.permutation(num_atoms) if perm_index is None else perm_index

    permuted_lines = [atom_lines[i] for i in perm_index]
    lines = lines[:2] + list(permuted_lines)  
    with open(tmp_file, 'w') as f:
        f.writelines(lines)
    return tmp_file, np.argsort(perm_index) # inverse permutation

def reverse_mat_permutation(M, atom_labels, inv_perm_index):
    """Reverse the permutation of a matrix.
    atom_labels: ao_labels() return for mol"""
    atom_ids = []
    pattern = re.compile(r'(\d+)')
    for lbl in atom_labels:
        match = pattern.search(lbl)
        if match:
            atom_ids.append(int(match.group(1)))
        else:
            raise ValueError(f"Keine Atom-ID in Label: {lbl}")
    unique_ids = sorted(set(atom_ids))
    start = 0
    atom_ranges = []
    for uid in unique_ids:
        count_id = atom_ids.count(uid)
        atom_ranges.append((start, start + count_id))
        start += count_id
    reordered_ranges = [atom_ranges[i] for i in inv_perm_index]
    reordered_indices = []
    for (s, e) in reordered_ranges:
        reordered_indices.extend(range(s, e))
    return M[np.ix_(reordered_indices, reordered_indices)]
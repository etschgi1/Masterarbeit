from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scf_guess_tools import calculate, load, Backend
from pyscf import scf
from scipy.linalg import eigh

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
    wf.kernel(dm0=density_guess)

    return {"cycles": wf.cycles, "conv": wf.converged, "summary": wf.scf_summary}

def plot_mat_comp(reference, prediction, reshape=False, title="Fock Matrix Comparison", ref_title="Reference", pred_title="Prediction", vmax=1.5):
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
    
    cbar = fig.colorbar(diff_plot, cax=ax[3])
    cbar.set_label("Difference Scale")
    
    plt.tight_layout()
    plt.show()
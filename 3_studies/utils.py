from collections import defaultdict
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scf_guess_tools import calculate, load, Backend
from pyscf import scf
from pyscf.gto import Mole
from scipy.linalg import eigh
import re
from pyscf.symm.Dmatrix import Dmatrix
from BlockMatrix import Block, BlockMatrix
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

def get_overlap(file, basis_set, symmetry=False, backend=Backend.PY): 
    lines = open(file).readlines()
    mol = load(file, backend)
    q = int(re.search(r"charge\s+(-?\d+)", lines[1]).group(1))
    m = int(re.search(r"multiplicity\s+(\d+)", lines[1]).group(1))
    mol = Mole(atom=file, charge=q, spin=m - 1, symmetry=symmetry)
    mol.basis = basis_set
    mol.build()
    return mol.intor('int1e_ovlp')

def load_mol(file, basis_set, symmetry=False, backend=Backend.PY):
    lines = open(file).readlines()
    q = int(re.search(r"charge\s+(-?\d+)", lines[1]).group(1))
    m = int(re.search(r"multiplicity\s+(\d+)", lines[1]).group(1))
    mol = Mole(atom=file, charge=q, spin=m - 1, symmetry=symmetry)
    mol.basis = basis_set
    mol.build()
    return mol


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

def reconstruct_Fock(diag, ovlp, K = 1.75): 
    """Take diagonal and reconstruct the Fock matrix using GWH
    """
    mat_dim = diag.shape[0]
    out = np.zeros((mat_dim, mat_dim))
    for i in range(mat_dim):
        for j in range(mat_dim):
            if i == j:
                out[i, j] = diag[i]
            else:
                out[i, j] = K * ovlp[i, j] * (diag[i] + diag[j]) / 2
    return out

def plot_fock_comparison(ex_test, ex_pred, size, matrix_metric="Fock", title="Fock Matrix Comparison", vmax=1.5):
    diff = ex_test - ex_pred
    rmse = root_mean_squared_error(ex_test, ex_pred)
    
    test_mat = unflatten_triang(ex_test, size) if ex_test.shape[0] != size else ex_test
    pred_mat = unflatten_triang(ex_pred, size) if ex_pred.shape[0] != size else ex_pred
    diff_mat = unflatten_triang(diff, size) if diff.shape[0] != size else diff
    
    fig, ax = plt.subplots(1, 4, figsize=(15, 5), width_ratios=[1, 1, 1, 0.1])
    fig.suptitle(f"{title}  |  RMSE: {rmse:.8f}")
    
    ax[0].imshow(test_mat, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax[0].set_title(f"{matrix_metric} converged (REFERENCE)")
    
    ax[1].imshow(pred_mat, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax[1].set_title(f"{matrix_metric} from overlap (PREDICTION)")
    
    diff_plot = ax[2].imshow(diff_mat, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax[2].set_title("Difference")
    
    cbar = fig.colorbar(diff_plot, cax=ax[3])
    cbar.set_label("Difference Scale")
    
    plt.tight_layout()
    plt.show()


def get_reordering(xyz_old, xyz_new): 
    """
    Get the reordering of the atoms in the new xyz xyz_file compared to the old one.
    Indices in list are the indices of the old xyz xyz_file! 
    """
    # read in the old and new xyz files
    with open(xyz_old, 'r') as f:
        old_lines = f.readlines()
    with open(xyz_new, 'r') as f:
        new_lines = f.readlines()
    
    nr_atoms = int(old_lines[0].strip())

    # get the atom names and coordinates from the old and new xyz files
    old_atoms = [line.split() for line in old_lines[2:nr_atoms+2]]
    new_atoms = [line.split() for line in new_lines[2:nr_atoms+2]]
    
    # get the reordering of the atoms
    reordering = [old_atoms.index(atom) for atom in new_atoms]
    assert len(reordering) == len(new_atoms), "Reordering length does not match new atom length"
    assert len(set(reordering)) == len(reordering), "Reordering contains duplicate indices"
    return reordering

def reorder_matrix(mat: np.ndarray,
                   reordering: list[int],
                   mol) -> np.ndarray:
    """
    Reorder an AO-based square matrix in atom blocks.

    Parameters
    ----------
    mat : (M, M) array
        The original AO matrix (M = total number of AOs).
    reordering : length-N_atoms list
        Each entry j is the index of the *old* atom that should
        move to new atom-position j.
    mol : pyscf.gto.Mole
        A PySCF molecule, used to get `mol.ao_labels()` and thus
        recover how many AOs live on each atom in the old ordering.

    Returns
    -------
    mat_new : (M, M) array
        The same matrix, but with atom-blocks permuted according
        to `reordering`.
    """
    # 1) Group AO indices by old-atom
    labels = mol.ao_labels()  # e.g. ["2 O 1s", "2 O 2s", "3 C 1s", ...]
    old2aos: dict[int, list[int]] = {}
    for ao_idx, lab in enumerate(labels):
        atom_idx = int(lab.split()[0])   # PySCF labels are 1-based
        old2aos.setdefault(atom_idx, []).append(ao_idx)

    # 2) Build new AO ordering by flattening blocks in the new atom order
    new_order = []
    for new_atom in range(len(reordering)):
        old_atom = reordering[new_atom]
        new_order.extend(old2aos[old_atom])

    # 3) Permute both axes
    mat_new = mat[np.ix_(new_order, new_order)]
    return mat_new

def reorder_Matrix_using_xyz_perm(mat: np.ndarray,
                                  xyz_old: str,
                                  xyz_new: str,
                                  mol) -> np.ndarray:
        """Reorder an AO-based square matrix in atom blocks
        using the permutation of the atoms in the xyz files.
        Parameters
        ----------
        mat : (M, M) array
            The original AO matrix (M = total number of AOs).
        xyz_old : str
        xyz_new : str
        mol : pyscf.gto.Mole
            THE OLD MOL (loaded with xyz_old and correct basis!), PySCF molecule, used to get `mol.ao_labels()`
        """
        reordering = get_reordering(xyz_old, xyz_new)
        return reorder_matrix(mat, reordering, mol)



def quaternion_to_euler_zyz(q): 
    """Convert a quaternion to Euler angles (Z-Y-Z convention)."""

    w, x, y, z = q.w, q.x, q.y, q.z
    
    phi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    theta = np.arcsin(2 * (w * y - z * x))
    psi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    return phi, theta, psi

def quaternion_from_axis_angle(axis, angle):
    """Generate a unit quaternion from an axis and angle."""
    half_angle = angle / 2
    s = np.sin(half_angle)
    c = np.cos(half_angle)
    return np.quaternion(c, *(axis * s))


def rotate_M(mol, axis, angle, M, maxL=3):
    """
    Rotate a Block Matrix M using the Wigner-D matrices for the given axis and angle.
    The rotation is applied to the blocks of M according to their angular momentum l.
    mol: PySCF Mole object
    axis: rotation axis as a 3D vector (numpy array)
    angle: rotation angle in radians
    M: BlockMatrix instance to be rotated
    maxL: maximum angular momentum to consider (default is 3 for s, p, d, f)
    """
    if not isinstance(M, BlockMatrix):
        raise TypeError("M must be a BlockMatrix instance")
    
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
    ao_lbls = mol.ao_labels(fmt=False, base=0)
    shells = defaultdict(list)
    for mu, (iatom, _, shell_name, _) in enumerate(ao_lbls):
        shells[(iatom, shell_name)].append(mu)

    
    # Unit‐quaternion for the axis‐angle rotation - try out euler to use pyscf's methods to stay consistent
    alpha, beta, gamma = quaternion_to_euler_zyz(quaternion_from_axis_angle(axis, angle))
    Dls = {l: Dmatrix(l, alpha, beta, gamma, reorder_p=True) for l in range(maxL)} 

    blocks_orig = M.blocks
    outM = M.copy()
    blocks_out = outM.blocks
    for key, block in blocks_orig.items(): # perform transformations block by block in sub-blocks (according to l)
        rows, cols = np.array(block.ls[0]), np.array(block.ls[1]) # l's of the rows and columns
        A = np.array(block.numpy, dtype=float)

        subblocks = {}
        for li in np.unique(rows):        # e.g. 0 then 1
            row_idx = np.where(rows == li)[0]
            for lj in np.unique(cols):    # e.g. 0 then 1
                if lj[-1] == 's' and li[-1] == 's': # skip s overlaps - no transformation
                    continue
                col_idx = np.where(cols == lj)[0]
                # this picks out the (li,lj) sub‐block
                sub = A[np.ix_(row_idx, col_idx)]
                subblocks[(li, lj)] = (row_idx, col_idx, sub)
        # transform and write back
        for (li,lj), (row_idx, col_idx, sub) in subblocks.items():
            # sanity check: len(idxs) should == 2l+1
            li, lj = l_map[li[-1]], l_map[lj[-1]]   # '2p'[-1] → 'p' → 1
            if len(row_idx) != 2*li+1 or len(col_idx) != 2*lj+1:
                raise ValueError(f"Expected {2*li+1} AOs for shell {shell_name}, got {len(row_idx)}")
            # insert back into block
            A[np.ix_(row_idx, col_idx)] = Dls[li] @ sub @ Dls[lj].T
        # overwrite the block with the transformed one
        blocks_out[key]._replace(A)
    # resymmetrize the matrix
    i_u, j_u = np.triu_indices_from(outM.Matrix, k=1)
    outM.Matrix[j_u, i_u] = outM.Matrix[i_u, j_u]
    return outM

def find_repo_root(repo_name='Masterarbeit'):
    from pathlib import Path    
    p = Path.cwd()
    for d in (p, *p.parents):
        if d.name == repo_name:
            return d
    raise FileNotFoundError(f"Could not find a folder named {repo_name} in {p} or its parents")
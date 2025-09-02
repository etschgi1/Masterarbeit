import numpy as np
from scf_guess_tools import Backend, cache, load, calculate
from collections import defaultdict
import quaternion
from pyscf.symm.Dmatrix import Dmatrix
from BlockMatrix import BlockMatrix, Block

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
from collections import defaultdict

def quaternion_to_euler_zyz(q): 
    """
    Convert a quaternion to Euler angles (Z-Y-Z convention).
    """

    # Extract the components of the quaternion
    w, x, y, z = q.w, q.x, q.y, q.z
    
    # Calculate the Euler angles
    phi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    theta = np.arcsin(2 * (w * y - z * x))
    psi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    return phi, theta, psi

def quaternion_from_axis_angle(axis, angle):
    half_angle = angle / 2
    s = np.sin(half_angle)
    c = np.cos(half_angle)
    return np.quaternion(c, *(axis * s))

def rotate_points(points, axis, angle):
    """
    Rotates points around a given axis by a specified angle.

    Args:
        points (np.ndarray): shape (N, 3)
        axis (np.ndarray): rotation axis
        angle (float): rotation angle in radians

    Returns:
        np.ndarray: rotated points
    """
    q = np.quaternion(np.cos(angle / 2), *(axis * np.sin(angle / 2)))
    q_conjugate = q.conjugate()
    rotated_points = []
    for point in points:
        p = np.quaternion(0, *point)
        rotated_point = q * p * q_conjugate
        rotated_points.append(rotated_point.imag)
    return np.array(rotated_points)


def rotate_M(mol, axis, angle, M, maxL=4, return_numpy=True):
    """
    Rotate a Block Matrix M using the Wigner-D matrices for the given axis and angle.
    The rotation is applied to the blocks of M according to their angular momentum l.
    mol: PySCF Mole object
    axis: rotation axis as a 3D vector (numpy array)
    angle: rotation angle in radians
    M: BlockMatrix instance to be rotated
    maxL: maximum angular momentum to consider (default is 4 for s, p, d, f)
    """
    if not isinstance(M, BlockMatrix):
        try:
            M = BlockMatrix(mol, Matrix=M)
        except: 
            raise TypeError("M must be a BlockMatrix instance")
    
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4} # we hopefully don't need higher than g
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
    if return_numpy:
        return outM.Matrix
    return outM


def rotated_xyz_content(xyz_source, new_coords): 
    """Returns a list of xyz file lines with the coordinates of xyz_source rotated to new_coords."""
    with open(xyz_source, 'r') as f:
        lines = f.readlines()
    xyz_source_nr_atoms = int(lines[0].strip())
    if len(new_coords) != xyz_source_nr_atoms:
        raise ValueError(f"Number of atoms in new_coords ({len(new_coords)}) does not match xyz_source ({xyz_source_nr_atoms})")
    for i, line in enumerate(lines[2:2+xyz_source_nr_atoms]):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Line {i+3} in xyz file does not have enough parts: {line}")
        parts[1:4] = new_coords[i]
        lines[i+2] = '\t'.join(map(str, parts))
        lines[i+2] += '\n' if not lines[i+2].endswith('\n') else ''
    return lines

def plot_mat_comp(reference, prediction, reshape=False, title="Fock Matrix Comparison", ref_title="Reference", pred_title="Prediction", vmax=1.5, labels1=None, labels2=None):
    import matplotlib.pyplot as plt
    diff = reference - prediction
    rmse = np.sqrt(np.mean((reference - prediction)**2))
    
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

def diis_rmse(overlap, density, fock): 
    """Eq 2.3 - Milacher"""
    E = fock @ density @ overlap - overlap @ density @ fock
    diis_rmse_ = np.sqrt(np.linalg.norm(E, ord='fro')**2 / (density.shape[0]**2))
    return diis_rmse_

def energy_elec(fock, density, coreH): 
    return np.trace((fock+coreH) @ density)

def energy_err(e_pred, e_conv): 
    return e_conv - e_pred, e_pred/e_conv -1
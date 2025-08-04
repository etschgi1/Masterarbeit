import numpy as np

def calc_quaratic_cmat(target, positions, masses):
    cmat = np.zeros((4, 4))
    p = target + positions
    m = target - positions
    cmat[0, 0] = (masses*(m[:, 0]**2 + m[:, 1]**2 + m[:, 2]**2)).sum()
    cmat[0, 1] = (masses*(p[:, 1]*m[:, 2] - m[:, 1]*p[:, 2])).sum()
    cmat[0, 2] = (masses*(m[:, 0]*p[:, 2] - p[:, 0]*m[:, 2])).sum()
    cmat[0, 3] = (masses*(p[:, 0]*m[:, 1] - m[:, 0]*p[:, 1])).sum()
    cmat[1, 1] = (masses*(m[:, 0]**2 + p[:, 1]**2 + p[:, 2]**2)).sum()
    cmat[1, 2] = (masses*(m[:, 0]*m[:, 1] - p[:, 0]*p[:, 1])).sum()
    cmat[1, 3] = (masses*(m[:, 0]*m[:, 2] - p[:, 0]*p[:, 2])).sum()
    cmat[2, 2] = (masses*(p[:, 0]**2 + m[:, 1]**2 + p[:, 2]**2)).sum()
    cmat[2, 3] = (masses*(m[:, 1]*m[:, 2] - p[:, 1]*p[:, 2])).sum()
    cmat[3, 3] = (masses*(p[:, 0]**2 + p[:, 1]**2 + m[:, 2]**2)).sum()
    cmat[1,0] = cmat[0,1]
    cmat[2,0] = cmat[0,2]
    cmat[3,0] = cmat[0,3]
    cmat[2,1] = cmat[1,2]
    cmat[3,1] = cmat[1,3]
    cmat[3,2] = cmat[2,3]
    return cmat

def calc_eckart_rotmat(target, positions, masses):
    cmat = calc_quaratic_cmat(target, positions, masses)
    v = np.linalg.eigh(cmat)[1][:, 0]
    rotate=np.zeros((3,3))
    rotate[0,0] = v[0]**2 + v[1]**2 - v[2]**2 - v[3]**2
    rotate[0,1] = 2.0*(v[1]*v[2] - v[0]*v[3])
    rotate[0,2] = 2.0*(v[1]*v[3] + v[0]*v[2])
    rotate[1,0] = 2.0*(v[1]*v[2] + v[0]*v[3])
    rotate[1,1] = v[0]**2 - v[1]**2 + v[2]**2 - v[3]**2
    rotate[1,2] = 2.0*(v[2]*v[3] - v[0]*v[1])
    rotate[2,0] = 2.0*(v[1]*v[3] - v[0]*v[2])
    rotate[2,1] = 2.0*(v[2]*v[3] + v[0]*v[1])
    rotate[2,2] = v[0]**2 - v[1]**2 - v[2]**2 + v[3]**2
    return rotate

def check_eckart_conditions(target, positions, masses):
    rr = np.cross(target, positions) * masses[:, None]
    r = rr.sum(axis=0)
    return np.abs(r).max()

def get_eckart_rotate(refatoms, atoms, centered=False, return_atoms = False):
    if centered:
        pos_ref = refatoms.positions
        pos = atoms.positions
    else:
        pos_ref = refatoms.positions-refatoms.get_center_of_mass()
        pos = atoms.positions-atoms.get_center_of_mass()
    masses = atoms.get_masses()
    rotate = calc_eckart_rotmat(pos_ref, pos, masses)
    return rotate

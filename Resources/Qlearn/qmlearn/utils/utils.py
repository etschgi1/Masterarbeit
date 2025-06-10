import numpy as np
import hashlib
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from pyscf import df,dft,gto
from pyscf.dft import numint

def dict_update(d, u):
    import collections.abc
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def copy_array(arr, out = None, add = False, index = None):
    if index is None : index = slice(None)
    if out is None :
        out = arr.copy()
    elif add :
        out[index] += arr
    else :
        out[index] = arr
    return out

def get_hash(x):
    value = hashlib.md5(np.array(sorted(x))).hexdigest()
    return value

def matrix_deviation(mat1, mat2):
    diff = np.abs(mat1 - mat2)
    errorsum = np.sum(diff)
    return errorsum

def array2blocks(arr, sections):
    blocks={}
    ix=0
    ns=len(sections)
    for i in range(ns):
        iy=0
        for j in range(ns):
            indices = np.s_[ix:ix+sections[i],iy:iy+sections[j]]
            blocks[(i,j)] = arr[indices]
            iy=iy+sections[j]
        ix=ix+sections[i]
    return blocks

def blocks2array(blocks, indices):
    new_blocks=[]
    for i in indices:
        la=[]
        for j in indices:
            la.append(blocks[(i,j)])
        new_blocks.append(la)
    arr = np.block(new_blocks)
    return arr

def wedge(a, b=None):
    if b is None: b = a
    ab1 = np.einsum('pq,rs->pqrs',a,b) + np.einsum('pq,rs->rspq',a,b)
    ab2 = np.einsum('pq,rs->prsq',a,b) + np.einsum('pq,rs->sqpr',a,b)
    return (ab1-ab2)*0.25

def unitary_decompose(amat, a=None, trace=None, r=None, identity=None):
    if a is None: a = np.trace(amat)
    if trace is None: trace = np.einsum('iijj->', amat)
    if r is None: r = amat.shape[0]
    if identity is None: identity = np.eye(r)
    A0 = 2.0 * trace / (r*(r-1)) * wedge(identity)
    A1_0 = 4.0 / (r-2) * wedge(a, identity)
    A1 = A1_0 - 2*(r-1)/(r-2) * A0
    A2 = amat - A1 - A0
    return A0, A1, A2

def ase_2_pyscf(mol):
    mol_stra=[]
    e=0
    while e < len(mol.get_chemical_symbols()):
          mol_stra.append(str(mol.get_chemical_symbols()[e])+
                            ' '+str(mol.get_positions()[e][0])+
                            ' '+str(mol.get_positions()[e][1])+
                            ' '+str(mol.get_positions()[e][2]))
          e+=1
    return mol_stra

def density_fitting_g2(mol,gamma2,auxmol):
    saux = auxmol.intor('int1e_ovlp')
    invsaux = np.linalg.inv(saux)
    ints_3c1e = df.incore.aux_e2(mol, auxmol, intor='int3c1e') # <kappa|ij>
    gamma2_df = np.einsum('mnst,mnQ,QA,stP,PB->AB', gamma2, ints_3c1e,invsaux,ints_3c1e,invsaux,optimize=True)
    return gamma2_df

def fft_gamma2(q,mol,gamma2,r12,ao_value,width,auxmol):
    b_2_a=0.529177
    g2_df = density_fitting_g2(mol,gamma2,auxmol)
    iqr=np.exp(1j*np.einsum('i,jki->jk',q,r12*b_2_a,optimize=True))
    return np.einsum('rs,AB,rA,sB->',iqr,g2_df,ao_value,ao_value,optimize=True)*width**3*width**3

def total_scater_factor(atoms,gamma2,limit=50,level=4,space=0.5,basis='6-31g*'):
    b_2_a=0.529177
    mol = gto.Mole()
    mol.atom = ase_2_pyscf(atoms)
    mol.basis = basis
    mol.build()
    size=level
    width = space
    coords = []
    for ix in np.arange(-size, size, width):
          for iy in np.arange(-size, size, width):
              for iz in np.arange(-size, size, width):
                  coords.append((ix,iy,iz))
    coords = np.array(coords)
    auxbasis = 'aug-ccpv5z-jk-fit'
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    ao_value = numint.eval_ao(auxmol, coords)
    shape=ao_value.shape[0]
    r12=np.zeros((shape,shape,3))
    for p in np.arange(shape):
          r12[:,p]=coords-coords[p]
    fft = []
    h_ = []
    for iz in range(limit):
          h=0.3*iz*b_2_a
          print(h)
          q=np.array([0.0,0.0,h])
          a=fft_gamma2(q,mol,gamma2,r12,ao_value,width,auxmol)
          q=np.array([0.0,h,0.0])
          b=fft_gamma2(q,mol,gamma2,r12,ao_value,width,auxmol)
          q=np.array([h,0.0,0.0])
          c=fft_gamma2(q,mol,gamma2,r12,ao_value,width,auxmol)
          fft.append((a+b+c)/3+mol.nelectron)
          h_.append(h)
    return np.array(h_),np.real(np.array(fft))

def fft_gamma(q,gamma,r,w,ao_value):
    b_2_a=0.529177 
    iqr=np.exp(1j*np.einsum('i,pi->p',q,r*b_2_a))
    mat = np.einsum('mn, p, pm, pn->',gamma, iqr*w, ao_value, ao_value, optimize=True)
    return mat

def elastic_scater_factor(atoms,gamma,limit=50,level=4,basis='6-31g*'):
    b_2_a = 0.529177
    mol = gto.Mole()
    mol.atom = ase_2_pyscf(atoms)
    mol.basis = basis
    mol.build()
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    r = grids.coords
    w = grids.weights
    ao_value = numint.eval_ao(mol,r)
    fft = []
    h_ = []
    for iz in range(limit):
          h=0.3*iz*b_2_a
          print(h)
          q=np.array([0.0,0.0,h])
          a=fft_gamma(q,gamma,r,w,ao_value)
          q=np.array([0.0,h,0.0])
          b=fft_gamma(q,gamma,r,w,ao_value)
          q=np.array([h,0.0,0.0])
          c=fft_gamma(q,gamma,r,w,ao_value)
          fft.append((a+b+c)/3)
          h_.append(h)

    fft_f = np.array(np.abs(np.real(fft))**2)
    return np.array(h_),fft_f


import numpy as np
from ase.geometry import get_distances_derivatives, get_distances

class FixBondLComb:
    """This is similar to ASE FixBondLengths, but with linear combination of bond lengths.
    sum_i(bond_length_i * coefs_i) = constant

     Parameters
    ----------
    pairs : array
        Array of pairs of atoms index
    coefs : array
        Array of pair bond length weights
    dt : float
        time steps in ASE Velocity Verlet integration
    tol : float
        If difference between weighted sum of bondlengths and fixed bond length is less than `tol`
        calculation is stoped.
    target: float
        Fixed Bondlength value
    maxiter : int
        maximum number of iteration
    """
    def __init__(self, pairs = None, coefs = None, dt = None, tol=1e-6, target=None, maxiter = 1000, scale = 2.0):
        self.pairs = np.array(pairs)
        if coefs is not None :
            self.coefs = np.array(coefs)
        else :
            self.coefs = np.ones(len(self.pairs))
        self.tol = tol
        self.maxiter = maxiter
        self.target = target
        self.scale = scale
        self.lm = 0.0
        self._dt = dt
        self.bm_nsteps = -1
        self.iter = 0

    def copy(self):
        kwargs = {
                'pairs' : self.pairs,
                'coefs' : self.coefs,
                'dt' : self.dt,
                'tol' : self.tol,
                'target' : self.target,
                'maxiter' : self.maxiter,
                'scale' : self.scale,
                }
        obj = self.__class__(**kwargs)
        return obj

    @property
    def dt(self):
        """time steps in ASE Velocity Verlet integration"""
        if self._dt is None :
            raise AttributeError("Please set the 'dt' firstly.")
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    def get_removed_dof(self, atoms):
        return 1

    def adjust_positions(self, atoms, new):
        """Keep adjusting atomic positions while keeping
           weighted sum of fix bond length constant

        Parameters
        ----------
        atoms : ASE atom object

        new : array
            positions of atoms which will change until converged

        Raises
        ------
        RuntimeError
            If calculation is not converged within max number of iteration
            a RuntimeError will raised
        """
        masses = atoms.get_masses()[:, None]

        if self.target is None:
            self.target = self.get_prims(atoms, atoms.positions)

        bmat0 = self.get_jacobian(atoms, atoms.positions)
        lm = 0.0
        lmd = np.zeros_like(new)
        for i in range(self.maxiter):
            value = self.get_prims(atoms, new)
            bmat = self.get_jacobian(atoms, new)
            diff = value - self.target
            if abs(diff) < self.tol : break
            dlm = diff / np.sum(bmat0*bmat/masses) / self.scale
            lm += dlm
            new -= dlm/masses*bmat
            lmd -= dlm/masses*bmat
        else:
            raise RuntimeError('RATTLE algorithm not converge')
        #-----------------------------------------------------------------------
        self.bm_nsteps = i
        self.bm_xi = value
        self.bm_lambda = 2.0*lm/(self.dt**2)
        self.iter += 1
        #-----------------------------------------------------------------------
        # Due to ASE use Velocity Verlet integration
        p = atoms.get_momenta() + lmd/self.dt
        atoms.set_momenta(p, apply_constraint = False)
        #-----------------------------------------------------------------------

    def adjust_momenta(self, atoms, p):
        """Update momentum of atoms using velocity verlet algorithm"""
        masses = atoms.get_masses()[:, None]
        vel = p / masses

        if self.target is None:
            self.target = self.get_prims(atoms, atoms.positions)

        bmat = self.get_jacobian(atoms, atoms.positions)
        for i in range(self.maxiter):
            diff = self.get_prims_vel(atoms, vel, bmat)
            if abs(diff) < self.tol : break
            dlm = diff / np.sum(bmat*bmat/masses)
            vel -= dlm/masses*bmat
        else:
            raise RuntimeError('RATTLE algorithm not converge')
        # print(f'bm_b converged at {i} step.', flush = True)
        # print(f'bm_xi_b : {diff}', flush = True)
        p[:] = vel*masses

    def adjust_forces(self, atoms, forces):
        pass

    def get_prims(self, atoms, pos):
        """Calculate weighted bond length of paired atoms

        Parameters
        ----------
        atoms : ASE atoms object

        pos : array
            positions of atoms

        Returns
        -------
        float
            weighted sum of paired bond length
        """
        value = 0.0
        for coef, ip in zip(self.coefs, self.pairs) :
            _, r = get_distances(pos[ip[0]], pos[ip[1]], cell = atoms.cell, pbc = atoms.pbc)
            value += coef*r[0][0]
        return value

    def get_prims_vel(self, atoms, vel, jacobian):
        value = 0.0
        for ip in self.pairs :
            value += np.sum(vel[ip[0]]*jacobian[ip[0]])
            value += np.sum(vel[ip[1]]*jacobian[ip[1]])
        return value

    def get_jacobian(self, atoms, pos, jacobian = None):
        """Calculate the Jacobian of positions of the paired atoms.

        Returns
        -------
        array
            Jacobian of postion of paired atoms
        """
        n = 2 # bond
        vectors = [pos[j] - pos[i] for i, j in self.pairs]
        derivs = get_distances_derivatives(vectors, cell=atoms.cell, pbc=atoms.pbc)
        if jacobian is None : jacobian = np.zeros_like(pos)
        for i, ip in enumerate(self.pairs):
            for j in range(n):
                jacobian[ip[j]] += derivs[i, j]*self.coefs[i]
        return jacobian

    def output(self, write = True):
        if self.bm_nsteps < 0 : return ''
        fstr = f'bm_nsteps_pos({self.iter}) converged at {self.bm_nsteps} step.\n'
        fstr+= f'bm_xi_pos({self.iter}) : {self.bm_xi} {self.target}\n'
        fstr+= f'bm_lambda_pos({self.iter}) : {self.bm_lambda}'
        if write :
            print(fstr, flush = True)
        return fstr
        # masses = atoms.get_masses()
        # bmat = self.get_jacobian(atoms, atoms.positions)
        # zet = np.sum(bmat*bmat/masses)
        # # 'lambda', '|z|^(-1/2)', 'kTG'

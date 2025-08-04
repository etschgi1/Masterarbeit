import os
import numpy as np
from ase import Atoms, io
import multiprocessing

from qmlearn.drivers.core import Engine
methods_psi4 = {
        'hf' : 'rhf+hf',
        'rhf' : 'rhf+hf',
        'scf' : 'rks+scf',
        'rks' : 'rks+scf',
        'mp2' : 'rhf+mp2',
        }

class EnginePsi4(Engine):
    r""" PySCF calculator

    Attributes
    ----------

    vext : ndarray
        External Potential.
    gamma : ndarray
        1-body reduced density matrix (1-RDM).
    etotal : float
        Total electronic energy.
    ovlp : ndarray
        Overlap Matrix.
    kop : ndarray
        Kinetic Energy Operator.
    nelectron : int
        Total number of electrons.
    xcfunc : str
        Exchange correlation Functional.
    eri : ndarray
        2-electron integrals: 8-fold or 4-fold ERIs or complex integral array with N^4 elements
        (N is the number of orbitals).
    """
    def __init__(self, **kwargs):
        import psi4
        self.psi4 = psi4
        self._xcfunc = None
        super().__init__(**kwargs)

    def init(self, **kwargs):
        mol = self.options.get('mol', None)
        mints = self.options.get('mints', None)
        basis = self.options.get('basis', '6-31g')
        xc = self.options.get('xc', None)
        verbose = self.options.get('verbose', 0)
        self.method = self.options.get('method', 'rks').lower()
        e_convergence = self.options.get('e_convergence', -np.log10(self.options.get('convergence', 1e-6)))
        d_convergence = self.options.get('d_convergence', -np.log10(self.options.get('convergence', 1e-6)))
        #
        nthreads = self.options.get('nthreads', multiprocessing.cpu_count())
        self.psi4.set_num_threads(nthreads)
        #
        if isinstance(self.method, str):
            self.method = methods_psi4.get(self.method, '')
        if self.method.count('+') != 1 :
            raise AttributeError("Sorry, only support two methods at the same time.")
        for fm in self.method.split('+') :
            if fm not in methods_psi4:
                raise AttributeError(f"Sorry, not support the {fm} method now")
        if mints is None :
            mol = self.init_mol(mol, basis=basis)
            basisset = self.psi4.core.BasisSet.build(mol, key='BASIS', other=basis)
            mints = self.psi4.core.MintsHelper(basisset)
        self.psi4.set_options({
            'reference': self.method.split('+')[0],
            'basis': basis,
            'e_convergence': int(e_convergence),
            'd_convergence': int(d_convergence),
            }, verbose = verbose)
        self.psi4_method = self.method.split('+')[1]
        self.mints = mints
        self.mol = self.mints.basisset().molecule()
        self.xc = xc

    def init_mol(self, mol, basis):
        ase_atoms = None
        if isinstance(mol, Atoms):
            ase_atoms = mol
        if isinstance(mol, (str, os.PathLike)):
            ase_atoms = io.read(mol)
        else:
            atom = mol
        if ase_atoms is not None :
            smol = ['{}\t{:.15f}\t{:.15f}\t{:.15f}'.format(atom.symbol, *atom.position) for atom in ase_atoms]
            smol = '\n'.join(smol)
            atom = self.psi4.geometry(smol)

        if isinstance(atom, self.psi4.core.Molecule):
            mol = atom
        else :
            mol = self.psi4.geometry(atom)
        return mol

    def run(self, ao_repr = True, **kwargs):
        r"""Caculate electronic properties using Psi4.

        Parameters
        ----------
        etotal : Total electronic energy
        wfn : Wavefunction of the system
        gamma : 1-RDM
        """
        etotal, wfn = self.psi4.energy(self.psi4_method, molecule=self.mol, dft_functional = self.xc, return_wfn=True)
        self.gamma = wfn.Da_subset('AO').np + wfn.Db_subset('AO').np
        self.wfn = wfn
        self.etotal = etotal

    @property
    def vext(self):
        if self._vext is None:
            self._vext = self.mints.ao_potential().np
        return self._vext

    @property
    def kop(self):
        if self._kop is None:
            self._kop = self.mints.ao_kinetic().np
        return self._kop

    @property
    def ovlp(self):
        if self._ovlp is None:
            self._ovlp = self.mints.ao_overlap().np
        return self._ovlp

    @property
    def eri(self):
        r"""2-electron integrals: 8-fold or 4-fold ERIs or complex integral array with N^4 elements (N is the number of orbitals).
        """
        if self._eri is None:
            self._eri = self.mints.ao_eri().np
        return self._eri

    @property
    def nelectron(self):
        r""" Total number of electrons. """
        charges = [self.mol.charge(i) for i in range(self.mol.natom())]
        return np.sum(charges)

    def calc_gamma(self, orb=None, occs=None):
	r"""Perform gamma calculation from PySCF

	Parameters
	----------
	orb: None
	   Default is to use value defined elsewhere in drivers
	occs: None
	   Default is to use value defined elsewhere in drivers
	"""

        raise AttributeError("Sorry, not implemented yet.")

    @property
    def xcfunc(self):
        r""" Function to define Exchange Correlation functional if it is not given.

        Parameters
        ----------
        gamma : ndarray
            1-RDM
        """
        if self._xcfunc is None:
            build_superfunctional = self.psi4.procrouting.dft.superfunctionals.build_superfunctional
            method = self.method.split('+')[0]
            if method in ['rks', 'uks'] :
                if method == 'rks' :
                    rks = True
                    ru = 'RV'
                else :
                    rks = False
                    ru = 'UV'
                xcfunctional = build_superfunctional(self.xc, rks)[0]
                self._xcfunc = self.psi4.core.VBase.build(self.mints.basisset(), xcfunctional, ru)
                self._xcfunc.initialize()
        return self._xcfunc

    def calc_exc(self, gamma):
        r""" Function to calculate the Exchange Correlation energy.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        exc : float
            Exchane Correlation energy.
        """
        # only for restricted formalism
        if self.xcfunc is not None :
            D = self.psi4.core.Matrix.from_array(gamma/2.0)
            self._xcfunc.set_D([D])
            V_out = self.psi4.core.Matrix.from_array(np.empty(gamma.shape))
            self._xcfunc.compute_V([V_out])
            exc=self._xcfunc.quadrature_values()["FUNCTIONAL"]
        else :
            eop = np.einsum('prqs,rs->pq', self.eri, gamma)
            exc = -np.einsum('pq,pq->', eop, gamma)*0.25
        return exc

    def calc_etotal(self, gamma, **kwargs):
        r""" Function to calculate Total Energy.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        etotal : float
            Total energy.
        """
        # only for restricted formalism
        e_nuc = self.mol.nuclear_repulsion_energy()
        jop = np.einsum('pqrs,rs->pq', self.eri, gamma)*0.5
        hop2 = self.kop + self.vext
        fop = hop2 + jop
        e1 = np.einsum('pq,pq->', fop, gamma)
        exc = self.calc_exc(gamma)
        etotal = e1 + exc + e_nuc
        return etotal

    def calc_dipole(self, gamma, **kwargs):
        r""" Get the total dipole moment.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        dip : ndarray
            The dipole moment on x, y and z component.
        """
        nucdip = self.mol.nuclear_dipole()
        ao_dipole = self.mints.ao_dipole()
        dip = np.zeros(3)
        for i in range(3):
            dip[i] = np.einsum('ij,ji->', ao_dipole[i].np, gamma)
            dip[i] += nucdip[i]
        return dip

    def calc_quadrupole(self, gamma, traceless = True):
        r"""  Function to calculate the total quadruple for XX, XY, XY, YY, YZ, ZZ components.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        quadp : ndarray
            An array containing a quadruple per component.
        """
        # XX, XY, XZ, YY, YZ, ZZ
        quadrupole = self.mints.ao_quadrupole()
        quadp = np.zeros(6)
        for i in range(6):
            quadp[i] = np.einsum('ji,ij->', quadrupole[i].np, gamma)
        quadp += self.calc_quadrupole_nuclear()
        if traceless :
            trace = (quadp[0] + quadp[3] + quadp[5])/3.0
            quadp[0] -= trace
            quadp[3] -= trace
            quadp[5] -= trace
        return quadp

    def calc_quadrupole_nuclear(self, mol = None):
        r""" Function to calculate the nuclear quadruple for XX, XY, XY, YY, YZ, ZZ components.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        quadrupol : ndarray
           An array containing the nuclear quadruple per component."""
        mol = mol or self.mol
        pos = mol.geometry().np
        charges = [mol.charge(i) for i in range(self.mol.natom())]
        charges = np.asarray(charges)
        nucquad = np.zeros(6)
        nucquad[0] = np.sum(charges * pos[:, 0] * pos[:, 0])
        nucquad[1] = np.sum(charges * pos[:, 0] * pos[:, 1])
        nucquad[2] = np.sum(charges * pos[:, 0] * pos[:, 2])
        nucquad[3] = np.sum(charges * pos[:, 1] * pos[:, 1])
        nucquad[4] = np.sum(charges * pos[:, 1] * pos[:, 2])
        nucquad[5] = np.sum(charges * pos[:, 2] * pos[:, 2])
        return nucquad

    def _get_loc_orb(self, nocc=-1):
        pass

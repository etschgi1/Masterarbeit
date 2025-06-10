import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr


class QMLCalculator(Calculator):
    r""" Mean QML calculator

    Attributes
    ----------
    qmmodel : QMMol object
        Reference QMMol object

    method : str
        Options

        | 'gamma' : Use QMLearn learning proccess to predict the desire property.
        | 'engine' : Use PySCF engine to predict the desire property.

    properties : list:str
        Options

        | 'energy' : Energy
        | 'forces' : Forces
        | 'dipole' : Dipole
        | 'stress' : Stress
        | 'gamma' : 1-RDM

    """
    implemented_properties = ['energy', 'forces', 'dipole', 'stress', 'gamma']

    def __init__(self, qmmodel = None, second_learn = {}, method = 'gamma',
            label='QMLearn', atoms=None, directory='.', refqmmol = None, properties = ('energy', ),
            **kwargs):
        Calculator.__init__(self, label = label, atoms = atoms, directory = directory, **kwargs)
        self.qmmodel = qmmodel
        self.second_learn = second_learn
        self.method = method
        self._refqmmol = refqmmol
        self._properties = properties

    @property
    def refqmmol(self):
        r"""QMMol object. """
        if self._refqmmol is None :
            if hasattr(self.qmmodel, 'refqmmol'):
                return self.qmmodel.refqmmol
            else :
                # qmmodel is refqmmol
                return self.qmmodel
        return self._refqmmol

    @refqmmol.setter
    def refqmmol(self, value):
        self._refqmmol = value

    @property
    def properties(self):
        if not isinstance(self._properties, set):
            if isinstance(self._properties, str):
                self._properties = {self._properties,}
            else :
                self._properties = set(self._properties)
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = value

    def calculate(self, atoms=None, properties=('energy', ), system_changes=all_changes):
        r""" Function to calculate the desire properties.

        Parameters
        ----------
        properties : list:str
            Options

            | Energy : 'energy'
            | Forces : 'forces'
            | Dipole : 'dipole'
            | Stress : 'stress'
            | 1-RDM : 'gamma'
        """
        properties = set(properties) | self.properties
        Calculator.calculate(self,atoms=atoms,properties=properties,system_changes=system_changes)
        atoms = atoms or self.atoms
        self.results['stress'] = np.zeros(6)
        if self.method == 'engine' :
            qmmol = self.refqmmol.duplicate(atoms, refatoms=atoms)
            self.calc_with_engine(qmmol, properties=properties)
        else :
            if self.method == 'gamma' :
                qmmol = self.refqmmol.duplicate(atoms.copy())
                self.calc_with_gamma(qmmol, properties=properties)
            else :
                raise AttributeError(f"Sorry, not support '{self.method}' now.")

    def calc_with_gamma(self, qmmol, properties = ['energy']):
        r""" Function to calculate the desire properties using QMLearn learning process.

        Parameters
        ----------
        properties : list:str
            Options

            | Energy : 'energy'
            | Forces : 'forces'
            | Dipole : 'dipole'
            | Stress : 'stress'
            | 1-RDM : 'gamma'
        """
        shape = self.qmmodel.refqmmol.vext.shape
        gamma = self.qmmodel.predict(qmmol).reshape(shape)
        if self.qmmodel.purify_gamma :
            gamma = qmmol.purify_gamma(gamma)
        m2 = self.second_learn.get('gamma', None)
        if m2 :
            gamma2 = self.qmmodel.predict(gamma, method = m2).reshape(shape)
            if self.qmmodel.purify_gamma :
                gamma2 = qmmol.purify_gamma(gamma2)
        else :
            gamma2 = gamma

        if 'energy' in properties :
            m2 = self.second_learn.get('energy', None)
            if m2 :
                energy = self.qmmodel.predict(gamma, method=m2)
                self.results['energy'] = energy * Ha
            else :
                energy = qmmol.calc_etotal(gamma2)
                self.results['energy'] = energy * Ha

        if 'forces' in properties:
            m2 = self.second_learn.get('forces', None)
            if m2 :
                forces = self.qmmodel.predict(gamma, method=m2)
            else :
                forces = qmmol.calc_forces(gamma2)
            forces = self.qmmodel.convert_back(forces, prop='forces')
            self.results['forces'] = forces * Ha/Bohr
            # forces_shift = np.mean(self.results['forces'], axis = 0)
            # print('Forces shift :', forces_shift, flush = True)
            # self.results['forces'] -= forces_shift

        if 'dipole' in properties :
            m2 = self.second_learn.get('dipole', None)
            if m2 :
                dipole = self.qmmodel.predict(gamma, method=m2)
            else :
                dipole = qmmol.calc_dipole(gamma2)
            dipole = np.dot(dipole, qmmol.op_rotate_inv)
            self.results['dipole'] = dipole * Bohr

        if 'stress' in properties:
            self.results['stress'] = np.zeros(6)

        if 'gamma' in properties :
            gamma = self.qmmodel.convert_back(gamma2, prop='gamma')
            self.results['gamma'] = gamma

    def calc_with_engine(self, qmmol, properties = ('energy')):
        r""" Function to calculate the desire properties using PySCF engine.

        Parameters
        ----------
        properties : list:str
            Options

            | Energy : 'energy'
            | Forces : 'forces'
            | Dipole : 'dipole'
            | Stress : 'stress'
            | 1-RDM : 'gamma'
        """
        qmmol.engine.run(properties = properties)
        if 'energy' in properties :
            energy = qmmol.engine.etotal
            self.results['energy'] = energy * Ha
        if 'forces' in properties:
            forces = qmmol.engine.forces
            self.results['forces'] = forces * Ha/Bohr
        if 'stress' in properties:
            self.results['stress'] = np.zeros(6)
        if 'dipole' in properties :
            dipole = qmmol.calc_dipole(qmmol.engine.gamma)
            self.results['dipole'] = dipole * Bohr
        if 'gamma' in properties :
            self.results['gamma'] = qmmol.engine.gamma

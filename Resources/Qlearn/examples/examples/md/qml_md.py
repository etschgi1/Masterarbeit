import numpy as np
import sys
import ase
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, force_temperature
from ase.md.verlet import VelocityVerlet
from ase import units


from qmlearn.io.model import db2qmmodel
from qmlearn.api.api4ase import QMLCalculator


if __name__ == "__main__":
    #
    np.random.seed(8888)
    T = 300
    #
    dbfile = sys.argv[1]
    if len(sys.argv)>2:
        atoms = ase.io.read(sys.argv[2])
    else:
        atoms = ase.io.read('./opt.xyz')
    qmmodel = db2qmmodel(dbfile, names = '*')
    #
    second_learn = {
            'gamma' : 'd_gamma',
            'energy' : 'd_energy',
            'forces' : 'd_forces',
            }
    atoms.calc = QMLCalculator(qmmodel = qmmodel, second_learn = second_learn, method = 'gamma', properties=('dipole', ))
    # atoms.calc = QMLCalculator(qmmodel = qmmodel.refqmmol, method = 'engine', properties=('energy', ))
    MaxwellBoltzmannDistribution(atoms, temperature_K = T, force_temp=True)
    p = atoms.get_momenta()
    c = p.sum(axis = 0)
    p -= c/len(atoms)
    atoms.set_momenta(p)
    force_temperature(atoms, T)

    dyn = VelocityVerlet(atoms, timestep=0.5*units.fs, trajectory='md_nve.traj', logfile='md_nve.log')
    dyn.run(20)

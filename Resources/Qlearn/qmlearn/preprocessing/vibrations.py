import numpy as np
import itertools as it

def iter_vibrations_modes(atoms, modes, energies = None, temperature = 0.02585199101165164,
        nimages = 30, method = 'linear', cross = 1):

    if energies is not None :
        for i in range(len(modes)):
            modes[i] *= np.sqrt(temperature / np.abs(energies[i]))

    if method == 'cos' :
        phases = np.cos(np.linspace(0, np.pi, nimages))
    elif method == 'sin' :
        phases = np.sin(np.linspace(np.pi/2, np.pi*3/2, nimages))
    else :
        phases = np.linspace(-1, 1, nimages)

    if cross > 1 :
        modes_new = []
        for im in it.combinations(np.arange(len(modes)), cross):
            mode = None
            for i in im :
                if mode is None :
                    mode = modes[i]
                else :
                    mode = mode + modes[i]
            # mode = mode/np.sqrt(cross)
            modes_new.append(mode)
        modes = np.asarray(modes_new)

    for mode in modes :
        for s in phases :
            atoms_m = atoms.copy()
            atoms_m.positions += s * mode
            yield atoms_m

def get_vibrations_atoms(vib, indices = None, min_cross = 1, max_cross = 1, **kwargs):
    # vib is ase.vibrations
    if hasattr(vib, 'get_vibrations'): vib = vib.get_vibrations()
    if indices is None : indices = slice(None)
    if isinstance(indices, int) : indices = [indices]
    modes = vib.get_modes(all_atoms=True)[indices]
    energies = np.abs(vib.get_energies())[indices]
    atoms = vib.get_atoms()
    for cross in range(min_cross, max_cross + 1) :
        for atoms_m in iter_vibrations_modes(atoms, modes, energies = energies, cross = cross, **kwargs):
            yield atoms_m

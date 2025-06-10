import numpy as np
from ase import units
from qmlearn.drivers.rotate import get_eckart_rotate

def get_frame_dipoles(images, dipoles = None, return_images = False):
    #
    if dipoles is None or len(dipoles) < 1 :
        dipoles = []
        for atoms in images:
            dip = atoms.calc.get_dipole_moment()
            dipoles.append(dip)
    #
    refatoms = images[0]
    dip0 = dipoles[0]
    new = [dip0]
    for a, dip in zip(images[1:], dipoles[1:]):
        rotmat = get_eckart_rotate(refatoms, a)
        dip = np.dot(dip, rotmat)
        #
        if np.dot(dip, dip) > 1E-4 :
            if np.dot(dip, new[-1]) < 0.0: dip *= -1.0
        #
        new.append(dip)
    new = np.asarray(new)
    return new

def get_wavenumber(nsteps, timestep = 1.0, sigma = 500):
    dt = timestep * 1E-15
    t = np.arange(nsteps)*timestep
    wf = np.exp(-t**2/(sigma)**2)
    wavenumber = np.fft.rfftfreq(nsteps, dt*units._c*100)
    return wavenumber, wf

def dipoles2ir(dipoles, timestep = 1.0, sigma = 500, derivative = True, **kwargs):
    """calculate the IR from dipoles
    """
    wavenumber, wf = get_wavenumber(len(dipoles), timestep = timestep, sigma=sigma)
    #
    fts=[]
    for i in range(3):
        if derivative :
            ds = np.gradient(dipoles[:,i], edge_order=2)
            ds = ds/timestep
        else :
            ds = dipoles[:, i] - dipoles[0, i]
        #
        b = np.convolve(ds, ds[::-1], mode='same')[:len(ds)]*wf
        ft = np.fft.rfft(b, axis=0)
        fts.append(ft)
    fts=np.asarray(fts)
    intensity = fts.imag**2 + fts.real**2
    intensity = intensity.sum(axis=0)
    if not derivative : intensity *= wavenumber**2
    return wavenumber, intensity

def dipoles2ir_total(dipoles, timestep = 1.0, sigma = 500, derivative = True, **kwargs):
    """calculate the IR from dipoles
    """
    wavenumber, wf = get_wavenumber(len(dipoles), timestep = timestep, sigma=sigma)
    #
    ds = np.sqrt(np.sum(dipoles**2, axis=1))
    if derivative :
        ds = np.gradient(ds, edge_order=2)
        ds = ds/timestep
    else :
        ds = ds - ds[0]
    #
    b = np.convolve(ds, ds[::-1], mode='same')[:len(ds)]*wf
    ft = np.fft.rfft(b)
    intensity = ft.imag**2 + ft.real**2
    if not derivative : intensity *= wavenumber**2
    return wavenumber, intensity

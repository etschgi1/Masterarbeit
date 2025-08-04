import os
import numpy as np
from ase import Atoms, io
from qmlearn.io.hdf5 import DBHDF5
from qmlearn.utils import tenumerate

def read_images(traj, format=None, index=None):
    if index is None :
        inds = slice(None)
    else :
        inds = index
    #
    if isinstance(traj[0], Atoms):
        images = traj[inds]
    else :
        if not format :
            format = os.path.splitext(traj)[-1][1:]
        if format : format = format.lower()
        if format == 'traj' :
            from ase.io.trajectory import Trajectory
            images = Trajectory(traj)
            if index: images = images[inds]
        elif format in ['xyz', 'exyz', 'extxyz'] :
            images = io.read(traj, index = inds)[inds]
        else :
            raise AttributeError(f"Sorry, not support '{format}' format.")
    return images

def read_db(filename, names = '*'):
    db = DBHDF5(filename, 'r')
    if names is None : names = '*'
    if isinstance(names, str):
        method = names
        names = dict.fromkeys(['qmmol', 'atoms', 'properties'])
        names['qmmol'] = db.get_names(method + '/qmmol*')[0]
        names['atoms'] = db.get_names(method + '/train_atoms*')[0]
        names['properties'] = db.get_names(method + '/train_prop*')[0]
        print(f'Guess DB names : {names}', flush = True)
    data = {}
    for key in names :
        if key == 'qmmol' :
            data[key] = db.read_qmmol(names[key])
        elif key == 'atoms' :
            data[key] = db.read_images(names[key])
        elif key == 'properties' :
            data[key] = db.read_properties(names[key])
        elif not key.startswith('_') :
            raise ValueError(f'The key {key} can not read from the db')
    data['_names'] = names.copy()
    db.close()
    return data

def write_db(output, qmmol, images, properties, prefix = 'train', names = None, **kwargs):
    if names is None or len(names) < 2 : names = [None, ]*3
    db = DBHDF5(output, 'w', qmmol=qmmol, **kwargs)
    db.write_qmmol(qmmol, name = names[0], **kwargs)
    db.write_images(images, prefix=prefix, name = names[1], **kwargs)
    db.write_properties(properties, prefix=prefix, name = names[1], **kwargs)
    print(db.names, flush = True)
    db.close()

def merge_db(filenames, names = '*', output = None):
    for i, filename in tenumerate(filenames):
        data = read_db(filename, names=names)
        if i == 0 :
            qmmol = data['qmmol']
            images = data['atoms']
            properties = data['properties']
            for k in properties :
                properties[k] = [properties[k]]
            properties['dipole'][0][0, 0] = 1000.0
        else :
            images.extend(data['atoms'])
            for k in properties :
                if k not in data['properties'] :
                    raise AttributeError(f"The {filename} missing {k} property.")
                properties[k].append(data['properties'][k])
    #
    for k in properties :
        properties[k] = np.concatenate(properties[k])
    #
    data = {
            'qmmol' : qmmol,
            'atoms' : images,
            'properties' : properties,
            }
    if output is not None :
        write_db(output, qmmol, images, properties, names = names)
    return data

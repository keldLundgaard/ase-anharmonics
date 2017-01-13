import numpy as np


def to_massweight_coor(mode, atoms, indices=None):
    """ Transform a mode to massweighted mode

    atoms : ase atoms object
        all atoms in molecule
    mode : numpy array
        mode with coordinates of all elements
    """

    if indices is None:
        assert (3*atoms.get_number_of_atoms() == len(mode)),\
            '3 times the number of atoms must equal number of modes'

        indices = range(len(mode)//3)

    m = atoms.get_masses()[indices]

    return mode*np.repeat(m**0.5, 3)


def to_none_massweight_coor(mode, atoms, indices=None):
    """ Transform a massweighted mode to none-massweighted mode"""
    if indices is None:
        assert len(atoms) == len(mode)/3, \
            "The masses to take are not properly defined"
        indices = range(len(mode)/3)

    m = atoms.get_masses()[indices]

    return np.repeat(m**(-0.5), 3)*mode

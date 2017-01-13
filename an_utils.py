import numpy as np


def to_massweight_coor(movement_vector, atoms, indices=None):
    """Transform a movement_vector from xyz coordinates to mass
    weighted coordinates.

    Args:
        movement_vector (numpy array): vector with coordinates of all
            elements or specific elements.

        atoms (ase object): ase atoms object

        indices[optional] (list): list of atomic indices that
            movement_vector describes.
    Returns:
        The mass weighted coordinates of the input
    """
    if indices is None:
        assert len(movement_vector) == 3*len(atoms), \
            'Unexpected length of movement_vector'

        indices = range(len(movement_vector)//3)

    m = atoms.get_masses()[indices]

    return movement_vector*np.repeat(m**0.5, 3)


def to_none_massweight_coor(movement_vector, atoms, indices=None):
    """Transform a mass weighted movement_vector to normal xyz coordinates.

    Args:
        movement_vector (numpy array): vector with coordinates of all
            elements or specific elements.

        atoms (ase object): ase atoms object

        indices[optional] (list): list of atomic indices that
            movement_vector describes.
    Returns:
        Movement_vector in normal xyz coordinates
    """
    if indices is None:
        assert len(movement_vector) == 3*len(atoms), \
            'Unexpected length of movement_vector'

        indices = range(len(movement_vector)//3)

    m = atoms.get_masses()[indices]

    return np.repeat(m**(-0.5), 3)*movement_vector

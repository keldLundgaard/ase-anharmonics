import numpy as np
from an_utils import to_massweight_coor


def get_trans_dict(from_atom_to_atom, vib):
    """Get the tangent of the translational mode"""
    indices = vib.indices
    atoms = vib.atoms
    positions = atoms.get_positions()

    # Movements from position 1 to 2
    delta = positions[from_atom_to_atom[1]]-positions[from_atom_to_atom[0]]

    # tangent to follow
    mode_tangent = np.hstack([delta for i in range(len(indices))])

    # normalize the tangent
    mode_tangent *= 1./np.sqrt(np.sum(mode_tangent**2))

    # Convert to mass weighted coordinates
    mode_tangent_mass_weighted = to_massweight_coor(mode_tangent,
                                                    atoms,
                                                    indices=indices)

    trans_mode_dict = {
        "type": 'translation',
        "from_atom_to_atom": from_atom_to_atom,
        "indices": indices,
        "mode_tangent": mode_tangent,
        "mode_tangent_mass_weighted": mode_tangent_mass_weighted,
        "mode_position_delta": delta,
    }

    return trans_mode_dict

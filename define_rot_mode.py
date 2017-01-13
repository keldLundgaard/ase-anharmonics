import numpy as np
from an_utils import to_massweight_coor


def get_rot_dict(atoms,
                 basepos,
                 branch,
                 symnumber=1,
                 indices=None,
                 rot_axis=None):
    """Define the rotational mode.
    Get rotational axis, mode vector, and moment of inertia
    """

    basepos_arr = np.array(basepos)
    branch_arr = np.array(branch)

    assert (np.all(branch_arr < atoms.get_number_of_atoms())), \
        '\nBad branch - contains higher atom index than available'

    # Finding rot axis
    COM = atoms[branch].get_center_of_mass()
    if rot_axis is None:  # 'Guess' rotation axis
        rot_axis_arr = basepos_arr-COM
    else:  # User specified rotation axis
        rot_axis_arr = np.array(rot_axis)

    axis_norm = np.linalg.norm(rot_axis_arr)
    assert (not np.allclose(axis_norm, 0, 1e-5, 1e-7)), \
        """\nCould not determine rotation axis, length of axis vector is
        Possible cause: center of mass is same as specified point"""
    rot_axis_arr = rot_axis_arr/axis_norm

    moment_of_intertia = get_moment_of_intertia(
        atoms[branch], basepos, rot_axis_arr)

    mode_tangent = calculate_rot_mode(
        atoms, basepos_arr, rot_axis_arr, branch_arr)

    # defining the normal mode only in the indices used for the
    # vibrational analysis
    if indices is not None:
        mode_tangent = mode_tangent.reshape((-1, 3))[indices].ravel()
    else:
        indices = range(len(atoms))

    mode_tangent_mass_weighted = to_massweight_coor(
        mode_tangent, atoms, indices=indices)

    return {
        'type': 'rotation',
        'symnumber': symnumber,
        'base_pos': basepos_arr,
        'branch': branch_arr,
        'rot_axis': rot_axis_arr,
        'mode_tangent': mode_tangent,
        'mode_tangent_mass_weighted': mode_tangent_mass_weighted,
        'inertia': moment_of_intertia,
        'indices': indices, }


def get_moment_of_intertia(atoms, x0, r_rot):
    """Returns the moment of intertia
    """
    I = 0.
    atom_p = atoms.get_positions()
    atom_m = atoms.get_masses()
    for ap, am in zip(atom_p, atom_m):
        I += sum(np.cross(r_rot, ap-x0)**2.)*am
    return I


def calculate_rot_mode(atoms, base_pos, rot_axis, branch_arr,
                       mass_weight=True, normalize=True):
    """ Calculates the rotation mode for specified branch.

    Each atom in the branch rotates around the given rotation axis,
      which is perpendicular to the rotation axis and the radial vector:
      v_rot = v_axis cross (B->A)
    v_rot are weighted by distance from center of axis
    v_rot are also transformed into mass-weighted coordinates

    """
    ap = atoms.get_positions()
    v_zero = np.zeros(3*len(ap))
    v_rot = v_zero.reshape(-1, 3)

    BC = rot_axis/np.linalg.norm(rot_axis)
    # For each atom find vector and weight to rotation axis
    for i in branch_arr:
        BA = np.array(ap[i])-np.array(base_pos)
        BA = BA - np.dot(BA, BC)
        v_rot[i] = np.cross(BC, BA)

    v_rot = np.ravel(v_rot)
    if mass_weight:
        v_rot = to_massweight_coor(v_rot, atoms)
    if normalize:
        v_rot = v_rot / np.linalg.norm(v_rot)
    return v_rot


def rotatepoints(rotationcenter, rotationaxis, angle, atompos):
    """
    Rotate some coordinates
    See http://en.wikipedia.org/wiki/Rotation_matrix

    Args:
        rotationcenter (numpy array): center for rotation
        rotationaxis (numpy array): axis to rotate around
        angle (float): angle to rotate in radians
        atompos (numpy array): positions to rotate
    """
    x0 = np.matrix(rotationcenter)
    u = np.matrix(rotationaxis)
    ap = np.matrix(atompos)

    natoms = ap.shape[0]
    rp = np.zeros([natoms, 3])

    R = np.matrix([
        [
            np.cos(angle)+u[0, 1]**2*(1.-np.cos(angle)),
            u[0, 1]*u[0, 1]*(1.-np.cos(angle))-u[0, 2]*np.sin(angle),
            u[0, 1]*u[0, 2]*(1.-np.cos(angle))+u[0, 1]*np.sin(angle)],
        [
            u[0, 1]*u[0, 0]*(1.-np.cos(angle))+u[0, 2]*np.sin(angle),
            np.cos(angle)+u[0, 1]**2*(1.-np.cos(angle)),
            u[0, 1]*u[0, 2]*(1.-np.cos(angle))-u[0, 0]*np.sin(angle)],
        [
            u[0, 2]*u[0, 0]*(1.-np.cos(angle))-u[0, 1]*np.sin(angle),
            u[0, 2]*u[0, 1]*(1.-np.cos(angle))+u[0, 0]*np.sin(angle),
            np.cos(angle)+u[0, 2]**2*(1.-np.cos(angle))]])

    # Repeat center coordinate natoms rows
    x0 = np.tile(x0, [natoms, 1])

    # Vectors from rot center for all atoms
    ap = ap - x0

    # Apply rotation transformation
    rp = np.dot(R, ap.T)

    # Adding the offset
    rp = rp.T + x0

    return rp

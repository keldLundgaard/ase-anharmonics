from copy import copy
import sys

import numpy as np

from ase.io.trajectory import Trajectory

from define_rot_mode import calculate_rot_mode, rotatepoints
from anh_base import BaseAnalysis


class RotAnalysis(BaseAnalysis):
    """Module for calculate the partition function of rotational modes!
    """
    def __init__(
        self,
        an_mode,
        atoms,
        an_filename=None,
        settings={},
        log=sys.stdout,
        verbosity=1,
    ):
        super(RotAnalysis, self).__init__()

        self.an_mode = an_mode
        self.atoms = atoms
        self.an_filename = an_filename
        self.settings = settings
        self.log = log
        self.verbosity = verbosity

        # Checks
        assert self.an_mode['type'] == 'rotation'

        # settings

        self.E_max_kT = settings.get('E_max_kT', 5)
        self.temperature = settings.get('temperature', 300)  # Kelvin
        self.use_forces = settings.get('use_forces', False)
        # Convergence tolorance
        self.rel_Z_mode_change_tol = settings.get('rel_Z_mode_tol', 0.005)

        self.initialize()

    def initial_sampling(self):
        """ Function to start initial sampling of the rotational
        mode. This can be done before extra samples are introduced.
        """

        # initializing
        if len(self.an_mode.get('displacements', [])) == 0:
            self.an_mode['displacements'] = self.get_initial_angles(
                nsamples=self.settings.get('rot_nsamples', 5))
            self.add_rot_energy(None)  # adding ground state

        # getting initial data points
        while (len(self.an_mode['displacements']) >
               len(self.an_mode.get('displacement_energies', []))):

            next_angle = self.an_mode['displacements'][
                len(self.an_mode['displacement_energies'])]

            self.add_rot_energy(next_angle)

    def sample_new_point(self):
        """What new angle to sample:

        We take the maximum angle distance between two samples scaled with
        the exponenital to the average potential energy of the two angles.
         > exp(avg(E[p0],E[p2])/kT)
        """
        sample_angles = list(self.an_mode['displacements'])
        angle_energies = list(copy(self.an_mode['displacement_energies']))

        angles_sort_args = np.argsort(sample_angles)

        angles = np.array([sample_angles[i] for i in angles_sort_args])
        energies = np.array([angle_energies[i] for i in angles_sort_args])
        energies -= np.min(energies)

        angle_spacings = [angles[i+1] - angles[i]
                          for i in range(len(angles)-1)]

        scaled_angle_spacings = [
            angle_spacings[i]*np.exp(-(energies[i]+energies[i+1])/(2*self.kT))
            for i in range(len(angles)-1)]

        arg = np.argmax(scaled_angle_spacings)
        new_angle = angles[arg]+angle_spacings[arg]/2.
        self.an_mode['displacements'] = list(
            np.hstack((self.an_mode['displacements'], new_angle)))

        self.add_rot_energy(new_angle)

    def add_rot_energy(self, angle):
        """ Add groundstate energy for a rotation by angle (input) to
        the current mode object.

        Args:
            angle (float): angle of the rotation in radians
        """
        if angle:  # it will otherwise do a groundstate calculation
            # It should use the initial groundstate energy if the system
            # is in a position similar to the starting point.
            # We thereby save a DFT calculation as the old is reused.
            if (np.abs(2.*np.pi/self.an_mode['symnumber']-angle) > 1e-5):
                new_positions = self.get_rotate_positions(angle)
                self.atoms.set_positions(new_positions)

        if not self.an_mode.get('displacement_energies'):
            self.an_mode['displacement_energies'] = list()

        if self.use_forces:
            e = self.atoms.get_potential_energy(force_consistent=True)

            # For the forces, we need the projection of the forces
            # on the normal mode of the rotation at the current angle
            v_force = self.atoms.get_forces()[
                self.an_mode['indices']].reshape(-1)

            mode = calculate_rot_mode(
                self.atoms,
                self.an_mode['base_pos'],
                self.an_mode['rot_axis'],
                self.an_mode['branch'],
                mass_weight=False,
                normalize=False).reshape((-1, 3))[
                    self.an_mode['indices']].ravel()

            f = float(np.dot(v_force, mode))

            if not self.an_mode.get('displacement_forces'):
                self.an_mode['displacement_forces'] = [f]
            else:
                self.an_mode['displacement_forces'].append(f)
        else:
            e = self.atoms.get_potential_energy()

        # adding to trajectory:
        if self.traj is not None:
            self.traj.write(self.atoms)

        self.an_mode['displacement_energies'].append(e)

        self.atoms.set_positions(self.groundstate_positions)

        # save to backup file:
        if self.an_filename:
            self.save_to_backup()

    def get_initial_angles(self, nsamples=5):
        """ Returns at which initial angles the energy calculations
        should be done.
        0 and 2pi is not necessary, as these are already included. """
        angles = (
            2. * np.pi
            / ((nsamples-1) * self.an_mode['symnumber'])
            * np.array(range(0, nsamples)))

        # Adding pertubations to break symmetries
        # Seed to make it consistent
        np.random.seed(1)
        angles += np.hstack([0, 5e-2 * np.random.random(nsamples-1)])
        return angles

    def get_rotate_positions(self, angle):
        """ Get the atomic positions of branch that are rotated
        after the branch has been rotated by angle.

        Args:
            angle (float): angle of rotation in radians
        returns:
            rot_pos (numpy array): the positions of the atoms in the
                branch.
        """

        rot_pos = copy(self.groundstate_positions)

        rot_pos[self.an_mode['branch']] = rotatepoints(
            self.an_mode['base_pos'],
            self.an_mode['rot_axis'],
            angle,
            rot_pos[self.an_mode['branch']])

        return rot_pos

    def make_inspection_traj(
            self,
            num_displacements=10,
            filename=None):
        """Make trajectory file for translational mode to inspect"""
        if filename is None:
            filename = self.an_filename+'_inspect.traj'

        traj = Trajectory(filename, mode='w', atoms=self.atoms)

        old_pos = self.atoms.positions.copy()
        calc = self.atoms.get_calculator()
        self.atoms.set_calculator()

        angles = self.get_initial_angles(nsamples=num_displacements)

        for angle in angles:
            new_pos = self.get_rotate_positions(angle)
            self.atoms.set_positions(new_pos)
            traj.write(self.atoms)
            self.atoms.set_positions(old_pos)

        self.atoms.set_calculator(calc)
        traj.close()

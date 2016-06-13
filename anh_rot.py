from copy import copy
import sys

import numpy as np

from ase.io.trajectory import Trajectory

from define_rotmode import calculate_rot_mode, rotatepoints
from anh_base import BaseAnalysis
from fit_rots import PeriodicFit
from fit_settings import fit_settings


class RotAnalysis(BaseAnalysis):
    """Module for calculate the partition function of rotational modes!
    """
    def __init__(
        self,
        an_mode,
        atoms,
        traj_filename=None,
        bak_filename=None,
        settings={},
        log=sys.stdout,
        verbosity=2,
    ):
        super(RotAnalysis, self).__init__()

        self.an_mode = an_mode
        self.atoms = atoms
        self.traj_filename = traj_filename
        self.bak_filename = bak_filename
        self.settings = settings
        self.log = log
        self.verbosity = verbosity

        # Checks
        assert self.an_mode['type'] == 'rotation'

        # settings
        self.fit_forces = settings.get('fit_forces', False)
        self.E_max_kT = settings.get('E_max_kT', 5)
        self.temperature = settings.get('temperature', 300)  # Kelvin
        self.use_force_consistent = settings.get('use_force_consistent', False)
        # Convergence tolorance
        self.rel_Z_mode_change_tol = settings.get('rel_Z_mode_tol', 0.005)

        self.initialize()

    def initial_sampling(self):
        # initializing
        if len(self.an_mode.get('rot_angles', [])) == 0:
            self.an_mode['rot_angles'] = self.get_initial_angles()
            self.add_rot_energy(None)  # adding ground state

        # getting initial data points
        while (len(self.an_mode['rot_angles']) >
               len(self.an_mode.get('rot_energies', []))):

            next_angle = self.an_mode['rot_angles'][
                len(self.an_mode['rot_energies'])]

            self.add_rot_energy(next_angle)

    def sample_until_convergence(self):
        # initialize history to check convergence on
        self.ZPE = []
        self.entropy_E = []

        # while not converged and samples < max-samples
        self.ZPE_hist = []
        self.Z_mode_hist = []

        while self.is_converged() is False:
            if len(self.ZPE_hist) > 0:
                # sample a new point
                self.sample_new_point()

            if self.verbosity > 1:
                self.log.write('Step %i \n' % len(self.ZPE_hist))

            # Fit mode
            fit_settings.update({
                'symnumber': self.an_mode['symnumber'],
                'verbose': False,
                'search_method': 'iterative',
            })

            fitobj = PeriodicFit(fit_settings)

            if self.fit_forces:
                fitobj.set_data(
                    self.an_mode['rot_angles'],
                    self.an_mode['rot_energies'],
                    self.an_mode.get('rot_forces', []))
            else:
                fitobj.set_data(
                    self.an_mode['rot_angles'],
                    self.an_mode['rot_energies'],
                    [])

            fitobj.run()

            ZPE, Z_mode, energies = self.get_thermo(fitobj)

            self.ZPE_hist.append(ZPE)
            self.Z_mode_hist.append(Z_mode)

        return ZPE, Z_mode, energies

    def sample_new_point(self):
        """What new angle to sample:

        We take the maximum angle distance between two samples scaled with
        the exponenital to the average potential energy of the two angles.
         > exp(avg(E[p0],E[p2])/kT)
        """
        # Here we need a good algorightm for choosing the next point to sample

        self.an_mode['rot_angles']

        sample_angles = list(self.an_mode['rot_angles'])
        angle_energies = list(copy(self.an_mode['rot_energies']))

        # adding the point for making it fully periodic so that points can be
        # added in between
        angle_energies.append(angle_energies[0])
        sample_angles.append(2.*np.pi/self.an_mode['symnumber'])

        angles_sort_args = np.argsort(sample_angles)

        angles = np.array([sample_angles[i] for i in angles_sort_args])
        energies = np.array([angle_energies[i] for i in angles_sort_args])
        energies -= -np.min(energies)

        angle_spacings = [angles[i+1] - angles[i]
                          for i in range(len(angles)-1)]

        scaled_angle_spacings = [
            angle_spacings[i]*np.exp(-(energies[i]+energies[i+1])/(2*self.kT))
            for i in range(len(angles)-1)]

        arg = np.argmax(scaled_angle_spacings)
        new_angle = angles[arg]+angle_spacings[arg]/2.
        self.an_mode['rot_angles'] = np.hstack((
            self.an_mode['rot_angles'], new_angle))

        self.add_rot_energy(new_angle)

    def add_rot_energy(self, angle):
        if angle:  # it will otherwise do a groundstate calculation
            new_positions = self.get_rotate_positions(angle)
            self.atoms.set_positions(new_positions)

        if not self.an_mode.get('rot_energies'):
            self.an_mode['rot_energies'] = list()

        if self.use_force_consistent:
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

            if not self.an_mode.get('rot_forces'):
                self.an_mode['rot_forces'] = [f]
            else:
                self.an_mode['rot_forces'].append(f)
        else:
            e = self.atoms.get_potential_energy()

        # adding to trajectory:
        if self.traj is not None:
            self.traj.write(self.atoms)

        self.an_mode['rot_energies'].append(e)

        self.atoms.set_positions(self.groundstate_positions)

        # save to backup file:
        if self.bak_filename:
            self.save_to_backup()
        return

    def get_initial_angles(self, nsamples=5):
        """ Returns at which initial angles the energy calculations
        should be done.
        0 and 2pi is not necessary, as these are already included. """
        angles = (
            2.*np.pi /
            ((nsamples+1)*self.an_mode['symnumber'])
            * np.array(range(0, nsamples+1)))
        return angles

    def get_rotate_positions(self, angle):
        rot_pos = copy(self.groundstate_positions)

        rot_pos[self.an_mode['branch']] = rotatepoints(
            self.an_mode['base_pos'],
            self.an_mode['rot_axis'],
            angle,
            rot_pos[self.an_mode['branch']])

        return rot_pos

    def make_rotation_traj(self, num_angles, filename='inspect_an_mode.traj'):
        """ make a rotational traj file to easily inspect the defined
        rotational mode
        """

        traj = Trajectory(filename, mode='w', atoms=self.atoms)

        old_pos = self.atoms.positions.copy()
        calc = self.atoms.get_calculator()
        self.atoms.set_calculator()

        angles = self.get_initial_angles(nsamples=num_angles)

        for angle in angles:
            new_pos = self.get_rotate_positions(angle)
            self.atoms.set_positions(new_pos)
            traj.write(self.atoms)
            self.atoms.set_positions(old_pos)

        self.atoms.set_calculator(calc)
        traj.close()

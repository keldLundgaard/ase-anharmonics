import sys
from copy import copy

import numpy as np

# from ase.parallel import paropen
# from ase.io.trajectory import Trajectory

from anh_base import BaseAnalysis
from fit_legendre import NonPeriodicFit
from fit_settings import fit_settings


class VibAnalysis(BaseAnalysis):
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
        super(VibAnalysis, self).__init__()

        self.an_mode = an_mode
        self.atoms = atoms
        self.traj_filename = traj_filename
        self.bak_filename = bak_filename
        self.settings = settings
        self.log = log
        self.verbosity = verbosity

        # Checks
        assert self.an_mode['type'] == 'vibration'

        # settings
        self.fit_forces = settings.get('fit_forces', False)
        self.E_max_kT = settings.get('E_max_kT', 5)
        self.temperature = settings.get('temperature', 300)  # Kelvin
        self.use_force_consistent = settings.get('use_force_consistent', False)
        self.rel_Z_mode_change_tol = settings.get('rel_Z_mode_tol', 0.005)

        self.initialize()

    def initial_sampling(self):
        # initializing
        self.mode_xyz = to_none_massweight_coor(
            self.an_mode['mode'],
            self.atoms,
            self.an_mode['indices'])

        if len(self.an_mode.get('displacements', [])) == 0:

            self.an_mode['displacements'] = [0.]  # Starting point
            self.add_displacement_energy(None)  # Groundstate energy

            self.an_mode['displacements'] = list(
                np.hstack([
                    0.,
                    self.get_initial_displacements()]))

        # getting initial data points
        while (len(self.an_mode['displacements']) >
               len(self.an_mode.get('displacement_energies', []))):

            next_displacement = self.an_mode['displacements'][
                len(self.an_mode['displacement_energies'])]

            self.add_displacement_energy(next_displacement)

    def get_initial_displacements(self):
        """Returning the displacements for initial sampling of the
        potential energy curve."""
        n_init_samples = self.settings.get('n_init_samples', 6)
        max_disp = self.settings.get('max_disp', 0.05)  # angstrom
        step_multi_kT_disp = self.settings.get('step_multi_kT_disp', 0.25)

        e_hnu = abs(self.an_mode['hnu'])
        # initially, we try to move out to where the energy is going up by kT
        step_kT = self.kT/e_hnu * step_multi_kT_disp

        assert n_init_samples % 2 == 0, "Only even number of samples allowed"

        # The maximum displacement of of the atoms in angstrom
        a_disp_length = np.max(np.linalg.norm(self.mode_xyz.reshape(-1, 3),
                                              axis=1))

        step_size = step_kT
        # If the step is too large then we use the max disp
        if step_size*a_disp_length > max_disp:
            step_size = max_disp/a_disp_length

        steps = np.linspace(0., step_size, n_init_samples/2+1)[1:]

        displacements = np.hstack((-steps[::-1], [0.], steps))

        return displacements

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
                # raise NotImplementedError
                # self.sample_new_point()
                pass

            if self.verbosity > 1:
                self.log.write('Step %i \n' % len(self.ZPE_hist))

            # Fit mode
            fit_settings.update({
                'verbose': False,
                'search_method': 'iterative',
            })

            fitobj = NonPeriodicFit(fit_settings)

            x_disp = np.array(self.an_mode['displacements'])
            e_target = np.array(self.an_mode['displacement_energies'])

            if self.fit_forces:
                fitobj.set_data(
                    x_disp,
                    e_target,
                    self.an_mode.get('displacement_forces', []))
            else:
                fitobj.set_data(
                    x_disp,
                    e_target,
                    [])

            fitobj.run()

            ZPE, Z_mode, energies = self.get_thermo(fitobj)
            self.ZPE_hist.append(ZPE)
            self.Z_mode_hist.append(Z_mode)
            self.fitobj = fitobj

        if self.settings.get('plot_energy_surface'):
            import matplotlib.pylab as plt

            N_fit = 200
            disp_range = np.max(x_disp) - np.min(x_disp)
            fit_range = np.linspace(np.min(x_disp)-disp_range/10.,
                                    np.max(x_disp)+disp_range/10., N_fit)
            fit_vals = [fitobj.fval(xi) for xi in fit_range]

            plt.plot(self.an_mode['displacements'],
                     np.array(self.an_mode['displacement_energies']),
                     'x', label='target')
            plt.plot(fit_range, fit_vals, '-', label='fit')

            fn = self.an_mode.get('info', {}).get('plot_fn')
            if fn:
                plt.savefig(fn)
            else:
                plt.show()

        return ZPE, Z_mode, energies

    def sample_new_point(self):
        """What new angle to sample:
        We take the maximum angle distance between two samples scaled with
        the exponenital to the average potential energy of the two angles.
         > exp(avg(E[p0],E[p2])/kT)
        """
        # Here we need a good algorightm for choosing the next point to sample

        # self.rot_mode['rot_angles']
        #
        # sample_angles = list(self.rot_mode['rot_angles'])
        # angle_energies = list(copy(self.rot_mode['rot_energies']))
        #
        # # adding the point for making it fully periodic so that points can be
        # # added in between
        # angle_energies.append(angle_energies[0])
        # sample_angles.append(2.*np.pi/self.rot_mode['symnumber'])
        #
        # angles_sort_args = np.argsort(sample_angles)
        #
        # angles = np.array([sample_angles[i] for i in angles_sort_args])
        # energies = np.array([angle_energies[i] for i in angles_sort_args])
        # energies -= -np.min(energies)
        #
        # angle_spacings = [angles[i+1] - angles[i]
        #                   for i in range(len(angles)-1)]
        #
        # scaled_angle_spacings = [
        #     angle_spacings[i]*np.exp(-(energies[i]+energies[i+1])
            # /(2*self.kT))
        #     for i in range(len(angles)-1)]
        #
        # arg = np.argmax(scaled_angle_spacings)
        # new_angle = angles[arg]+angle_spacings[arg]/2.
        # self.rot_mode['rot_angles'] = np.hstack((
        #     self.rot_mode['rot_angles'], new_angle))
        #
        # self.add_rot_energy(new_angle)

    def add_displacement_energy(self, displacement):
        # TODO: This function should be a part of base

        if displacement is not None:  # otherwise do a groundstate calculation
            new_positions = self.get_displacement_positions(displacement)
            self.atoms.set_positions(new_positions)

        if not self.an_mode.get('displacement_energies'):
            self.an_mode['displacement_energies'] = list()

        if self.use_force_consistent:
            e = self.atoms.get_potential_energy(force_consistent=True)

            # For the forces, we need the projection of the forces
            # on the normal mode of the rotation at the current angle
            v_force = self.atoms.get_forces()[
                self.an_mode['indices']].reshape(-1)

            f = float(np.dot(v_force, self.an_mode['mode']))

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
        if self.bak_filename:
            self.save_to_backup()
        return

    def get_displacement_positions(self, stepsize):
        """
        This function is where we define how to follow the given mode
        """
        pos = copy(self.groundstate_positions)
        pos[self.an_mode['indices']] += stepsize * self.mode_xyz.reshape(-1, 3)
        return pos


def to_none_massweight_coor(mode, atoms, indices=None):
    """ Transform a massweighted mode to none-massweighted mode"""
    if indices is None:
        assert len(atoms) == len(mode)/3, \
            "The masses to take are not properly defined"
        indices = range(len(mode)/3)

    m = atoms.get_masses()[indices]

    return np.repeat(m**(-0.5), 3)*mode

import sys
from copy import copy

import numpy as np

from ase.io.trajectory import Trajectory

from anh_base import BaseAnalysis
from fit_legendre import NonPeriodicFit
from fit_settings import fit_settings

from an_utils import to_none_massweight_coor

from scipy.optimize import minimize_scalar


class VibAnalysis(BaseAnalysis):
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
        super(VibAnalysis, self).__init__()

        self.an_mode = an_mode
        self.atoms = atoms
        self.an_filename = an_filename
        self.settings = settings
        self.log = log
        self.verbosity = verbosity

        # Checks
        assert self.an_mode['type'] == 'vibration'

        # settings
        self.min_sample_energy_kT = settings.get('min_sample_energy_kT', 3)
        self.temperature = settings.get('temperature', 300)  # Kelvin
        self.use_forces = settings.get('use_forces', False)
        self.rel_Z_mode_change_tol = settings.get('rel_Z_mode_tol', 0.005)

        # The maximum displacement of of the atoms in angstrom
        self.mode_xyz = to_none_massweight_coor(
            self.an_mode['mode'],
            self.atoms,
            self.an_mode['indices'])

        self.max_stepsize = (
            self.settings.get('max_disp', 0.05)  # angstrom
            / np.max(np.linalg.norm(self.mode_xyz.reshape(-1, 3), axis=1))
        )
        self.initialize()

    def initial_sampling(self):
        """Initial sampling"""

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

    def get_initial_displacements(
            self,
            displacements=2,
            max_kT=None,
            max_step=True):
        """Returning the displacements for initial sampling of the
        potential energy curve."""

        # The following check because of the way we chose the displacements
        assert displacements % 2 == 0, "Only even number of samples allowed"

        if max_kT is None:
            step_multi_kT_disp = self.settings.get('step_multi_kT_disp', 0.25)
        else:
            step_multi_kT_disp = max_kT

        # If the mode has a real eigenvalue then we can use it
        # to choose how big a step we should use.
        if np.real(self.an_mode['hnu']) > 1e-3:
            e_hnu = abs(self.an_mode['hnu'])

            # initially, we try to move out to where the energy is going
            # up by kT
            step_kT = self.kT/e_hnu * step_multi_kT_disp
            step_size = step_kT
            # If the step is too large then we use the max step
            if max_step:
                if step_size > self.max_stepsize:
                    step_size = self.max_stepsize
        else:
            # If not then we put the step_size to be very big so
            #
            step_size = self.max_stepsize

        steps = np.linspace(0., step_size, displacements/2+1)[1:]

        displacements = np.hstack((-steps[::-1], [0.], steps))

        return displacements

    def get_fit(self):
        # Fit mode
        fit_settings.update({
            'verbose': False,
            'search_method': 'iterative',
        })

        fitobj = NonPeriodicFit(fit_settings)

        fitobj.set_data(
            np.array(self.an_mode['displacements']),
            np.array(self.an_mode['displacement_energies']),
            self.an_mode.get('displacement_forces', []))

        fitobj.run()

        return fitobj

    def sample_until_convergence(self):
        # initialize history to check convergence on
        self.ZPE = []
        self.entropy_E = []

        # while not converged and samples < max-samples
        self.ZPE_hist = []
        self.Z_mode_hist = []
        self.energies_last = []

        while self.is_converged() is False:
            if len(self.ZPE_hist) > 0:
                self.sample_new_point()

            if self.verbosity > 1:
                self.log.write('Step %i \n' % len(self.ZPE_hist))

            fitobj = self.get_fit()

            ZPE, Z_mode, energies = self.get_thermo(fitobj)
            self.ZPE_hist.append(ZPE)
            self.Z_mode_hist.append(Z_mode)
            self.energie_hist = energies
            self.fitobj = fitobj

        if self.settings.get('plot_mode'):
            self.plot_potential_energy(fitobj=fitobj)

        return ZPE, Z_mode, energies

    def sample_new_point(self):
        """What new angle to sample:
        We take the maximum angle distance between two samples scaled with
        the exponenital to the average potential energy of the two angles.
         > exp(avg(E[p0],E[p2])/kT)

        """
        # Should we sample further out
        sample_energies = np.array(self.an_mode['displacement_energies'])
        x = self.an_mode['displacements']

        min_energy_sampling = self.kT * self.min_sample_energy_kT

        arg_min_x = np.argmin(sample_energies)

        # Need to go out to the bounds in both directions
        for k, direction in enumerate([1, -1]):
            displacement_i = [i for i, xi in enumerate(x)
                              if direction*(xi-x[arg_min_x]) > 0.]

            if len(displacement_i) > 0:
                # First we check for if we have gotten the potential defined
                # good enough, or if we need to sample further out
                displacement_energies = [sample_energies[i]
                                         for i in displacement_i]
                sampling_energy_span = (np.max(displacement_energies) -
                                        np.min(displacement_energies))

                # widen the window of samples
                # If the sample is good enough then we can continue
                if sampling_energy_span > min_energy_sampling:
                    continue

                # We want to sample the potential energy curve further out

                boundary = np.sort([x[i] for i in displacement_i])[[-1, 0][k]]

                # fit with current points
                fitobj = self.get_fit()

                # check if I should go to the end, i.e. will the function value
                # at max step be larger than than kt_sample_min. If so search
                # for the proper step length. Otherwise, move max step length

                max_displacement = boundary + direction*self.max_stepsize

                # Checking if we are up to the bounds
                if (
                        fitobj.fval(max_displacement) - np.min(sample_energies)
                        < min_energy_sampling):
                    next_displacement = max_displacement
                else:
                    # reordering to ensure that the second value is bigger
                    # than the first
                    bounds = [boundary,
                              boundary+direction*self.max_stepsize
                              ][::direction]

                    # Function that I want to optimize for
                    # Taking the absolute as I want it to be as close to
                    # min_energy_sampling as possible
                    opt_func = lambda displacement: np.abs(
                        fitobj.fval(displacement) - np.min(sample_energies)
                        - min_energy_sampling)

                    # Find the best displacement
                    # scipy.optimize.minimize_scalar
                    # find root:
                    res = minimize_scalar(opt_func,
                                          bounds=bounds,
                                          method='bounded')

                    next_displacement = res.x

                self.an_mode['displacements'].append(next_displacement)
                self.add_displacement_energy(next_displacement)
            else:
                # We are in a situation where the furthest point that we
                # sampled in this direction has the lowest energy.
                # This could be caused by sampling an imaginary frequency
                # mode.
                x_arg_sort = np.argsort(x)
                if direction == 1:
                    next_displacement = (
                        2*x[x_arg_sort[-1]]-x[x_arg_sort[-2]])
                else:
                    next_displacement = (
                        2*x[x_arg_sort[0]]-x[x_arg_sort[1]])

                self.an_mode['displacements'].append(next_displacement)
                self.add_displacement_energy(next_displacement)

        #
        # Find the next point to sample as the one that we think would
        # add the most information. The simple approach is here to
        # calculate the spacings between the different displacements
        # and scale the spacing by the the exponential energy that is expected
        # for that point.
        #
        fitobj = self.get_fit()

        displacements = np.sort(self.an_mode['displacements'])
        sort_args = np.argsort(self.an_mode['displacements'])
        energies = np.array([self.an_mode['displacement_energies'][i]
                             for i in sort_args])
        energies -= np.min(energies)  # subtracting the groundstate energy

        if self.settings.get('use_scaled_spacings', 1):
            scaled_spacings = [
                (displacements[i+1]-displacements[i])
                * np.exp(-(energies[i+1]+energies[i])/(2*self.kT))
                for i in range(len(energies)-1)]

            max_arg = np.argmax(np.array(scaled_spacings))
        else:
            spacings = [
                (displacements[i+1]-displacements[i])
                for i in range(len(energies)-1)]
            max_arg = np.argmax(np.array(spacings))

        next_displacement = (
            displacements[max_arg+1]+displacements[max_arg]) / 2.

        self.an_mode['displacements'].append(next_displacement)
        self.add_displacement_energy(next_displacement)

    def add_displacement_energy(self, displacement):

        if displacement is not None:  # otherwise do a groundstate calculation
            new_positions = self.get_displacement_positions(displacement)
            self.atoms.set_positions(new_positions)

        if not self.an_mode.get('displacement_energies'):
            self.an_mode['displacement_energies'] = list()

        if self.use_forces:
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
        if self.an_filename:
            self.save_to_backup()

    def get_displacement_positions(self, stepsize):
        """
        This function is where we define how to follow the given mode
        """
        pos = copy(self.groundstate_positions)
        pos[self.an_mode['indices']] += stepsize * self.mode_xyz.reshape(-1, 3)

        return pos

    def make_inspection_traj(self, points=10, filename=None):
        """Make trajectory file for the vibrational mode for inspection"""
        if filename is None:
            filename = self.an_filename+'_inspect.traj'

        traj = Trajectory(filename, mode='w', atoms=self.atoms)

        old_pos = self.atoms.positions.copy()
        calc = self.atoms.get_calculator()
        self.atoms.set_calculator()

        displacements = self.get_initial_displacements(displacements=points)

        for displacement in displacements:
            new_pos = self.get_displacement_positions(displacement)
            self.atoms.set_positions(new_pos)
            traj.write(self.atoms)
            self.atoms.set_positions(old_pos)

        self.atoms.set_calculator(calc)
        traj.close()

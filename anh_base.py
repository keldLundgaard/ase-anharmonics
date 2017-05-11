import abc
import os
import pickle
# import warnings
import numpy as np

import ase.units as units
from ase.parallel import paropen
from ase.io.trajectory import Trajectory

from fit_settings import fit_settings

from energy_spectrum_solver import energy_spectrum
from fit_periodic import PeriodicFit
from fit_legendre import NonPeriodicFit


class BaseAnalysis(object):
    """Base module for calculating the partition function for
    vibrations and rotations.
    """
    __metaclass__ = abc.ABCMeta

    def initialize(self):
        """Initialize the analysis module."""

        # Calculate kT to be used:
        self.kT = units.kB * self.settings.get('temperature', 300)  # Kelvin

        # Define groundstate_positions -- free as already calculated
        self.groundstate_positions = self.atoms.get_positions()

        # Making trajectory file for the mode:
        if isinstance(self.an_filename, str):
            filename = self.an_filename+".traj"
            if os.path.exists(filename) and os.path.getsize(filename):
                mode = 'a'
            else:
                mode = 'w'
            self.traj = Trajectory(filename, mode, self.atoms)
        else:
            self.traj = None

        self.E_max_kT = 5

    def run(self):
        """Function to run full analysis following specifications with
        defined modes.

        Returns:
            The mode object
        """
        # Checks if there is a backup and loads it to self.an_mode if so
        self.restore_backup()

        if not (self.an_mode.get('ZPE') and
                self.an_mode.get('Z_mode') and
                self.an_mode.get('energy_levels')):

            # Do initial sampling points -- depends on type of mode
            self.initial_sampling()

            # Keep iterating until the convergence critia is fulfilled
            ZPE, Z_mode, energies = self.sample_until_convergence()

            # Update the mode definition with the calculated information
            self.an_mode.update({
                'ZPE': ZPE,
                'Z_mode': Z_mode,
                'energy_levels': energies})

            if self.an_filename:
                self.save_to_backup()

        return self.an_mode

    def sample_until_convergence(self):
        """ Function will choose new points along the rotation
        to calculate groundstate of and terminates if the thermodynamical
        properties have converged for the mode.
        """
        # initialize history to check convergence on
        self.ZPE = []
        self.entropy_E = []

        # while not converged and samples < max-samples
        self.ZPE_mode_hist = []
        self.ZE_mode_est = []  # estimate of entrypy contribution of mode

        while self.is_converged() is False:
            iteration_num = len(self.ZPE_mode_hist)

            if len(self.ZPE_mode_hist) > 0:
                # sample a new point
                self.sample_new_point()

            if self.settings.get('verbosity', 0) > 1:
                self.log.write('Step %i \n' % iteration_num)

            fitobj = self.get_fit()

            ZPE_mode, Z_mode, energies = self.get_thermo(fitobj)

            self.ZE_mode_est.append(self.kT * np.log(Z_mode))
            self.ZPE_mode_hist.append(ZPE_mode)

            if self.settings.get('plot_mode_each_iteration'):
                self.plot_potential_energy(
                    fitobj=fitobj,
                    name_add='_%02d' % iteration_num)

            if self.settings.get('fit_plot_regu_curve_iterations'):
                fitobj.plot_regularization_curve(
                    name_add='_%02d' % iteration_num)

        if self.settings.get('plot_mode'):
            self.plot_potential_energy(fitobj=fitobj)

        if self.settings.get('fit_plot_regu_curve'):
            fitobj.plot_regularization_curve()

        return ZPE_mode, Z_mode, energies

    def get_fit(self):
        user_fit_settings = dict(
            (key[4:], val) for key, val in self.settings.items()
            if key[:4] == 'fit_')
        user_fit_settings.update({'an_name': self.an_filename})
        if self.an_mode['type'] == 'vibration':
            fit_settings.update({
                'verbose': False,
                'search_method': 'iterative'})
            fit_settings.update(user_fit_settings)
            fitobj = NonPeriodicFit(fit_settings)

        elif self.an_mode['type'] == 'rotation':
            fit_settings.update({
                'symnumber': self.an_mode['symnumber'],
                'search_method': 'iterative',
                'iteration': len(self.Z_mode_hist)})
            fit_settings.update(user_fit_settings)
            fitobj = PeriodicFit(fit_settings)

        elif self.an_mode['type'] == 'translation':
            fit_settings.update({
                'symnumber': 1,
                'verbose': False,
                'search_method': 'iterative',
            })
            fit_settings.update(user_fit_settings)
            fitobj = PeriodicFit(fit_settings)

        else:
            raise ValueError(" Unknown an_mode")

        # Get all settings with fit_ to input in fitting

        fitobj.set_data(
            self.an_mode['displacements'],
            self.an_mode['displacement_energies'],
            self.an_mode.get('displacement_forces', []))

        fitobj.run()

        return fitobj

    def restore_backup(self):
        """Restore the mode object from a backup. If there is a backup
        file then it will load this into the mode object.
        """
        backup_loaded = 0
        # Check if the filename is there
        if self.an_filename and os.path.exists(self.an_filename+'.pckl'):
            # Open backup file
            backup = pickle.load(paropen(self.an_filename+'.pckl', 'rb'))

            # check if the backup correspond to the defined mode
            for test_key in ['type']:
                assert backup[test_key] == self.an_mode[test_key]

            if backup['type'] == 'rotation':
                assert abs(backup['inertia'] - self.an_mode['inertia']) < 1e-6

            self.an_mode = backup

            backup_loaded = 1

        return backup_loaded

    def save_to_backup(self):
        """Save current mode object to a pickle file. """
        pickle.dump(self.an_mode,
                    paropen(self.an_filename+'.pckl', 'wb'))

    def get_thermo(self, fitobj):
        """Calculate thermodynamics of mode. Currently supporting
        vibrational modes and rotational modes.

        Args:
            fitobj (object): The fitting object

        Returns:
            ZPE (float): The zero point energy for the mode.
            Z_mode (float): The partition function for mode.
            energies_truncated (list): Energy levels of modes
                truncated to specific max energy.
        """

        # Calculating the energy modes differently depending on the type
        if self.an_mode['type'] == 'rotation':
            Hcoeff = units._hbar**2/(units._amu * units._e
                                     * self.an_mode['inertia']*1e-20)
            xmin = 0.
            xmax = xmin+2.*np.pi/self.an_mode['symnumber']

            groundstate_energy = min(self.an_mode['displacement_energies'])

        elif self.an_mode['type'] == 'vibration':
            Hcoeff = units._hbar**2 / (2.*units._amu * units._e * 1e-20)

            xmin = np.min(self.an_mode['displacements'])
            xmax = np.max(self.an_mode['displacements'])

            groundstate_energy = min(self.an_mode['displacement_energies'])

        elif self.an_mode['type'] == 'translation':

            xmin = np.min(self.an_mode['displacements'])
            xmax = np.max(self.an_mode['displacements'])

            Hcoeff = units._hbar**2/(2*units._amu * units._e * 1e-20)

            groundstate_energy = min(self.an_mode['displacement_energies'])

        else:
            raise ValueError("No other types are currently supported")

        # Calculating energy spectrum
        energies = energy_spectrum(
            xmin, xmax, fitobj.fval, Hcoeff,
            mode=self.settings.get('energy_solver_mode', 'fast'))

        # subtracting the groundstate energy
        energies -= groundstate_energy

        # The zero point energy is per definition the first accessible energy
        # level
        ZPE = energies[0]

        # Calculating the partition function for the mode:
        Z_mode = 0.
        energies_truncated = []
        for i, e in enumerate(energies):
            # Only use the energies below a certain treshold
            if e > ZPE+self.E_max_kT*self.kT and i > 2:
                break

            Z_mode += np.exp((-e+ZPE)/self.kT)
            energies_truncated.append(e)

        return ZPE, Z_mode, energies_truncated

    def is_converged(self):
        """Check if the calculation has converged.
        Returns:
            converged (Bool): If the mode has been converged or not
        """
        converged = False

        iterations = len(self.ZPE_mode_hist)

        conv_tol = self.settings.get('conv_tol', 0.001)
        if iterations > self.settings.get('min_step_iterations', 2):

            ZPE_mode_delta = np.abs(
                self.ZPE_mode_hist[-1] - self.ZPE_mode_hist[-2])

            ZE_mode_est_delta = np.abs(
                self.ZE_mode_est[-1] - self.ZE_mode_est[-2])

            if self.settings.get('verbosity', 0) > 0:
                self.log.write(
                    'Deltas ZPE: %.2e eV Entropy est: %.2e eV tol: %.1e \n'
                    % (ZPE_mode_delta, ZE_mode_est_delta, conv_tol))

            if max(ZPE_mode_delta, ZE_mode_est_delta) < conv_tol:
                converged = 1
                if self.settings.get('verbosity', 0) > 0:
                    self.log.write('>>> Converged! <<< \n')

            elif iterations > self.settings.get('max_step_iterations', 15):
                converged = True
                self.log.write(
                    'Exiting after %i iterations: Cannot converge properly'
                    % iterations)
                self.log.write(
                    'Energy change ZPE: %.2e Entropy est: %.2e  tol: %.2e' %
                    (ZPE_mode_delta, ZE_mode_est_delta, conv_tol))
            else:
                converged = False

        return converged

    def plot_potential_energy(
            self, fitobj=None, filename=None, name_add=''):
        """Plot function to help debugging and understanding the modes."""
        import matplotlib.pylab as plt

        if filename is None:
            filename = self.an_filename + name_add + '.png'

        x = self.an_mode['displacements']
        energies = self.an_mode['displacement_energies']
        forces = self.an_mode.get('displacement_forces', [])

        dx = np.abs(x[1]-x[0]) / 4
        if len(forces):
            for i, (xi, ei, fi) in enumerate(zip(x, energies, forces)):
                # TODO: Why negative?
                df = -1 * dx * fi
                plt.plot(
                    [xi-dx, xi+dx], [ei - df, ei + df],
                    '-', color='k')

        plt.plot(x, energies, 'x', label=('Samples (%i points)' % (len(x))))

        if fitobj is not None:
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = fitobj.fval(x_fit)

            plt.plot(
                x_fit, y_fit, '-',
                label='fit with '+str(fitobj.order)+' coefficients')

        plt.legend()
        if self.an_mode['type'] == 'rotation':
            plt.xlabel('Displacement (radians)')
        else:
            plt.xlabel('Displacement (angstrom)')
        plt.ylabel('Potential energy (eV)')

        plt.savefig(filename)
        plt.clf()

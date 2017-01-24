import abc
import os
import pickle
import warnings
import numpy as np

import ase.units as units
from ase.parallel import paropen
from ase.io.trajectory import Trajectory

from energy_spectrum_solver import energy_spectrum


class BaseAnalysis(object):
    """Base module for calculating the partition function for
    vibrations and rotations.
    """
    __metaclass__ = abc.ABCMeta

    def initialize(self):
        """Initialize the analysis module."""

        # Calculate kT to be used:
        self.kT = units.kB * self.temperature

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

        return self.an_mode

    def restore_backup(self):
        """Restore the mode object from a backup. If there is a backup
        file then it will load this into the mode object.
        """

        # Check if the filename is there
        if self.an_filename and os.path.exists(self.an_filename+'.pckl'):
            # Open backup file
            backup = pickle.load(paropen(self.an_filename+'.pckl', 'rb'))

            # check if the backup correspond to the defined mode
            for test_key in ['type']:
                assert backup[test_key] == self.an_mode[test_key]

            if backup['type'] == 'rotation':
                assert backup['inertia'] == self.an_mode['inertia']

            self.an_mode = backup

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
            # xmin = 0. + fitobj.get_potential_bottom()
            # is currently not implemented due to required dependency on scipy.
            xmin = 0.
            xmax = xmin+2.*np.pi/self.an_mode['symnumber']

            groundstate_energy = self.an_mode['rot_energies'][0]

        elif self.an_mode['type'] == 'vibration':
            Hcoeff = units._hbar**2 / (2.*units._amu * units._e * 1e-20)

            xmin = np.min(self.an_mode['displacements'])
            xmax = np.max(self.an_mode['displacements'])

            groundstate_energy = self.an_mode['displacement_energies'][0]

        elif self.an_mode['type'] == 'translation':
            warnings.warn("Translational analysis needs testing!")

            xmin = np.min(self.an_mode['displacements'])
            xmax = np.max(self.an_mode['displacements'])

            Hcoeff = units._hbar**2/(2*units._amu * units._e * 1e-20)

            groundstate_energy = self.an_mode['displacement_energies'][0]
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

        iterations = len(self.ZPE_hist)

        if iterations > 2:
            rel_Z_mode_change = np.abs(
                (self.Z_mode_hist[-1]-self.Z_mode_hist[-2])
                / self.Z_mode_hist[-2])

            if self.verbosity > 1:
                print('Iteration: ', iterations)
                print('rel Z_mode change', rel_Z_mode_change)

            if iterations > 10:
                converged = True
                print('Exiting after 10 iterations: Cannot converge properly')
                print('rel_Z_mode_change', rel_Z_mode_change)
            else:
                if np.abs(rel_Z_mode_change) < self.rel_Z_mode_change_tol:
                    converged = True

        return converged

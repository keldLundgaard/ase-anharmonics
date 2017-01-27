import sys

import numpy as np

from ase.io.trajectory import Trajectory
from ase.constraints import FixedLine, FixAtoms
from ase.optimize import QuasiNewton
from ase.visualize import view

from anh_base import BaseAnalysis
from fit_periodic import PeriodicFit
from fit_settings import fit_settings


class TransAnalysis(BaseAnalysis):
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
        super(TransAnalysis, self).__init__()

        self.an_mode = an_mode
        self.atoms = atoms
        self.an_filename = an_filename
        self.settings = settings
        self.log = log
        self.verbosity = verbosity

        # Checks
        assert self.an_mode['type'] == 'translation'

        # settings
        self.fit_forces = settings.get('fit_forces', False)
        self.E_max_kT = settings.get('E_max_kT', 5)
        self.temperature = settings.get('temperature', 300)  # Kelvin
        self.use_force_consistent = settings.get('use_force_consistent', False)
        # Convergence tolerance
        self.rel_Z_mode_change_tol = settings.get('rel_Z_mode_tol', 0.01)

        self.initialize()

    def initial_sampling(self):
        """Start initial sampling of the mode. This can be done before extra
        samples are introduced.
        """
        # initializing
        if len(self.an_mode.get('displacements', [])) == 0:
            self.an_mode['displacements'] = self.get_initial_points()
            self.add_displacement_energy(None)  # adding ground state

        while (len(self.an_mode['displacements']) >
               len(self.an_mode.get('displacement_energies', []))):

            displacement = self.an_mode['displacements'][
                len(self.an_mode['displacement_energies'])]

            self.add_displacement_energy(displacement)

    def sample_until_convergence(self):
        """ Function will choose new points along the rotation
        to calculate groundstate of and terminates if the thermodynamical
        properties have converged for the mode.
        """
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

            fit_settings.update({
                'symnumber': 1,
                'verbose': False,
                'search_method': 'iterative',
            })

            fitobj = PeriodicFit(fit_settings)

            if self.fit_forces:
                fitobj.set_data(
                    self.an_mode['displacements'],
                    self.an_mode['displacement_energies'],
                    self.an_mode.get('trans_forces', []))
            else:
                fitobj.set_data(
                    self.an_mode['displacements'],
                    self.an_mode['displacement_energies'],
                    [])

            fitobj.run()

            ZPE, Z_mode, energies = self.get_thermo(fitobj)

            self.ZPE_hist.append(ZPE)
            self.Z_mode_hist.append(Z_mode)

        if self.settings.get('plot_mode'):
            self.plot_potential_energy(fitobj=fitobj)

        return ZPE, Z_mode, energies

    def get_initial_points(self, nsamples=5):
        """Get the points to initially calculate the potential
        energies at.

        Returns:
            displacements (list): The displacements along the
                translational path
        """
        displacements = (
            self.an_mode['transition_path_length']
            * (np.array(range(0, nsamples)) / (nsamples-1)))
        return displacements

    def sample_new_point(self):
        """Decide what displacement to sample next

        We take the maximum angle distance between two samples scaled with
        the exponenital to the average potential energy of the two angles.
         > exp(avg(E[p0],E[p2])/kT)
        """

        displacements = list(self.an_mode['displacements'])
        displacement_energies = list(self.an_mode['displacement_energies'])

        sort_args = np.argsort(displacements)

        displacements_sorted = np.array([displacements[i] for i in sort_args])
        energies = np.array([displacement_energies[i] for i in sort_args])
        energies -= np.min(energies)

        displacements_spacings = [
            displacements_sorted[i+1] - displacements_sorted[i]
            for i in range(len(displacements_sorted)-1)]

        scaled_displacements_spacings = [
            displacements_spacings[i]*np.exp(
                -(energies[i]+energies[i+1])/(2*self.kT))
            for i in range(len(displacements)-1)]

        arg = np.argmax(scaled_displacements_spacings)
        # Pick the point in between the two displacements that is the biggest
        new_displacement = (displacements_sorted[arg]
                            + 0.5*displacements_spacings[arg])

        self.an_mode['displacements'] = list(
            np.hstack((displacements, new_displacement)))

        self.add_displacement_energy(new_displacement)

    def add_displacement_energy(self, displacement):
        """Add the groundstate energy for a displacements along the
        translational path, and adds it to an_mode['displacement_energies'].

        Args:
            displacement (float): How much to follow translational path.
        """

        # Will otherwise do a groundstate calculation at initial positions
        if displacement:
            if displacement != self.an_mode['transition_path_length']:
                self.atoms.set_positions(
                    self.get_translation_positions(displacement))

                # Do 1D optimization
                axis_relax = self.an_mode.get('relax_axis')
                if axis_relax:
                    c = []
                    for i in self.an_mode['indices']:
                        c.append(FixedLine(i, axis_relax))
                    # Fixing everything that is not the vibrating part
                    c.append(
                        FixAtoms(mask=[
                            i not in self.an_mode['indices']
                            for i in range(len(self.atoms))]))
                    self.atoms.set_constraint(c)

                    old = self.atoms.get_positions()

                    # Optimization
                    dyn = QuasiNewton(self.atoms)
                    dyn.run(fmax=0.05)
                    self.atoms.set_constraint([])

                    new = self.atoms.get_positions()
                    print(new-old)
                    # raise
                    # raise NotImplementedError(" Pending feature")

        if not self.an_mode.get('displacement_energies'):
            self.an_mode['displacement_energies'] = list()

        if self.use_force_consistent:
            e = self.atoms.get_potential_energy(force_consistent=True)

            # For the forces, we need the projection of the forces
            # on the normal mode of the rotation at the current angle
            v_force = self.atoms.get_forces()[
                self.an_mode['indices']].reshape(-1)

            f = float(np.dot(
                v_force, self.an_mode['mode_tangent']))

            if not self.an_mode.get('trans_forces'):
                self.an_mode['trans_forces'] = [f]
            else:
                self.an_mode['trans_forces'].append(f)
        else:
            e = self.atoms.get_potential_energy()

        if self.traj is not None:
            self.traj.write(self.atoms)

        self.an_mode['displacement_energies'].append(e)

        # adding to trajectory:
        if self.traj is not None:
            self.traj.write(self.atoms)

        self.atoms.set_positions(self.groundstate_positions)

        # save to backup file:
        if self.an_filename:
            self.save_to_backup()

    def get_translation_positions(self, displacement):
        """Calculate the new positions of the atoms with the vibrational
        system moving along a linear translational path by a displacements
        given as an input.

        Args:
            displacement (float): The displacement along the translational path

        Returns:
            positions (numpy array): The new positions of the atoms with the
                vibrational system moved along the translational path.
        """

        positions = self.atoms.get_positions()
        for index in self.an_mode['indices']:
            positions[index] += displacement*self.an_mode['mode_tangent']

        return positions

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

        displacements = self.get_initial_points(nsamples=num_displacements)

        for displacement in displacements:
            new_pos = self.get_translation_positions(displacement)
            self.atoms.set_positions(new_pos)
            traj.write(self.atoms)
            self.atoms.set_positions(old_pos)

        self.atoms.set_calculator(calc)
        traj.close()

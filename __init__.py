"""Anharmonic modes module for ASE

Docs follows Google's python styling guide:
http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Developed by
Keld Lundgaard -- keld.lundgaard@gmail.com

Supervised by Thomas Bligaard

Other contributers:
Thomas Nygaard
"""
from __future__ import division

import sys
import os
from copy import copy

import numpy as np

import ase.units as units
from ase.parallel import paropen

from define_rot_mode import get_rot_dict
from define_trans_mode import get_trans_dict
from anh_rot import RotAnalysis
from anh_vib import VibAnalysis
from anh_trans import TransAnalysis

import warnings


class AnharmonicModes:
    """Calculate anharmonic modes: vibrational and rotational.

    The anharmonic mode extends the harmonic approximation of the
        vibrations module in ASE.

    examples:
        # Example 1:
        # Rotational mode
        # We look at the rotational mode of CH3 on a Al surface
        # This cannot be properly treated with the harmonic
        # approximation, which leads to a wrong estimation
        # of the entropy and zero-point-energy.

        from ase.structure import molecule
        from ase.lattice.surface import fcc111, add_adsorbate
        from ase.optimize import QuasiNewton
        from ase.constraints import FixAtoms
        from ase.calculators.emt import EMT
        from ase.vibrations import Vibrations

        from __init__ import AnharmonicModes

        # Setup system
        slab = fcc111('Al', size=(2, 2, 2), vacuum=3.0)
        CH3 = molecule('CH3')
        add_adsorbate(slab, CH3, 2.5, 'ontop')

        # Setup slap
        constraint = FixAtoms(mask=[a.symbol == 'Al' for a in slab])
        slab.set_constraint(constraint)
        slab.set_calculator(EMT())

        # Relax structure
        dyn = QuasiNewton(slab, trajectory='QN_slab.traj')
        dyn.run(fmax=0.05)

        # Running vibrational analysis
        vib = Vibrations(slab, indices=[8, 9, 10, 11])
        vib.run()
        vib.summary()

        # Solving rotational mode
        print('\n >> Anharmonics <<\n')

        AM = AnharmonicModes(vibrations_object=vib)
        rot_mode = AM.define_rotation(
            basepos=[0., 0., -1.],
            branch=[9, 10, 11],
            symnumber=3)
        AM.run()
        AM.summary()

        # QED

        # Example 2:
        # Anharmonic vibration
        # This example shows how to determine the vibrational mode
        # of H2 with higher precision than harmonic mode.
        # The H2 mode is very strong, which means that there is a large
        # distance between the zero-point energy (ZPE) and the first excited
        # energy level. The 1D schrodinger equation is solved like
        # particle in a box, and the difference between ZPE and the first
        # first vibrational frequency will therefore not be correctly
        # calculated, unless we sample the potential energy curve up to a
        # high energy (many times KT).
        #

        from ase.structure import molecule
        from ase.optimize import QuasiNewton
        from ase.calculators.emt import EMT
        from ase.vibrations import Vibrations

        from __init__ import AnharmonicModes

        H2 = molecule('H2')
        H2.set_calculator(EMT())
        dyn = QuasiNewton(H2)
        dyn.run(fmax=0.05)

        vib = Vibrations(H2, indices=[0, 1])
        vib.run()
        vib.summary()

        print('\n >> Anharmonics 1000k sampling up to 0.25 kT<<\n')

        AM = AnharmonicModes(vib,
                             settings={
                                 'temperature': 1000,
                                 'max_disp': 1.,
                                 # 'plot_energy_surface': 1
                             })

        vib_mode = AM.define_vibration(mode_number=-1)

        AM.run()
        AM.summary()

        print('>> Anharmonics 1000k with initial estimated out to 1.0 kT<<')

        AM = AnharmonicModes(vib,
                             settings={
                                 'temperature': 1000,
                                 'step_multi_kT_disp': 1.0,
                                 'max_disp': 1.,
                                 # 'plot_energy_surface': 1
                             })

        vib_mode = AM.define_vibration(mode_number=-1, mode_settings={})
        AM.run()
        AM.summary()

    """
    def __init__(
        self,
        vibrations_object,
        settings={},
        log=sys.stdout,
        verbosity=1,
    ):
        """Initialization

        Args:
            vibrations_object (obj): The vibration object of ASE.
            settings (optional[dict]): Change default behavior. E.g.
                "tempurature": 400  # (400K instead of 300K).
            log (optional[str]): name of log file.
            verbosity (optional[int]): Change how much extra information is
                printed.
        """
        self.vib = vibrations_object
        self.settings = settings

        if isinstance(log, str):
            log = paropen(log, 'a')
        self.log = log

        self.atoms = self.vib.atoms
        self.an_modes = []
        self.h_modes = None  # harmonic modes
        self.h_freq = None  # harmonic modes
        self.reduced_h_modes = self.vib.modes  # For harmonic subspace

        self.kT = units.kB * self.settings.get('temperature', 300)  # kelvin
        self.ev__inv_cm = 0.01 * units._e / units._c / units._hplanck

    def define_rotation(
            self,
            basepos,
            branch,
            symnumber=1,
            rot_axis=None,
            mode_settings={}):
        """Define a rotation

        Args:
            basepos (array or list): position that the rotation vector is
                defined from.
            branch (list): The atomic indices for the branch that are rotated.
            symnumber (int): symmetry number of the rotation.
            rot_axis (array or list): Rotational axis used to rotate the
                branch.
            mode_settings (optional[dict]): settings to overwrite the main
                settings for this mode in mode analysis.

        Returns:
            A dictionary that defines the rotational mode
        """

        # check that the mode is defined within the vibrational space:
        for i in branch:
            assert i in self.vib.indices

        an_mode = get_rot_dict(self.vib.atoms,
                               basepos,
                               branch,
                               indices=self.vib.indices,
                               rot_axis=rot_axis)

        an_mode.update({'mode_settings': mode_settings})

        self.an_modes.append(an_mode)

        self.reduced_h_modes = np.delete(self.reduced_h_modes, 0, axis=0)

        return an_mode

    def define_vibration(
            self,
            mode_number=None,
            mode_vector=None,
            mode_settings={}):
        """Define an anharmonic vibrational mode

        Args:
            mode_number (optional[int]): The mode number from the vibrational
                analysis, that should be treated anharmonically.
            mode_vector(optional[array or list]): Define a vibrational mode
                that should be treated anharmanically by giving a vector that
                specifies this vibrations movement.
                CURRENTLY NOT IMPLEMENTED!
            mode_settings (optional[dict]): settings to overwrite the main
                settings for this mode in mode analysis.

        Returns:
            A dictionary that defines the anharmonic vibrational mode
        """
        # Either mode_number or mode_vector should be set
        assert (mode_number is None) ^ (mode_vector is None)
        an_mode = {
            'type': 'vibration',
        }
        if mode_vector is not None:
            raise NotImplementedError

        elif mode_number is not None:
            # Get the current harmonic modes
            post_modes = self.get_post_modes()

            an_mode.update({
                # mode tangent that defines the vibrational mode fully
                'mode_tangent_mass_weighted': post_modes[mode_number],
                # The mode that we have selected
                'mode': post_modes[mode_number],
                # This is added to allow for a sanity check
                'indices': self.vib.indices,
                # Calculate the energy of the mode we'll take out
                'hnu': self.calculate_post_h_freqs()[mode_number],
                # attach mode number for reference
                'mode_number': mode_number,
                # attach settings only for this mode
                'mode_settings': mode_settings,
            })

            # Deleting the mode that we will treat differently from
            # the harmonic mode space
            self.reduced_h_modes = np.delete(post_modes, mode_number, axis=0)

        self.an_modes.append(an_mode)

        return an_mode

    def define_translation(
            self,
            from_atom_to_atom=None,
            mode_settings={}):
        """Define an anharmonic vibrational mode

        Args:
            from_atom_to_atom (optional[list: [#1:int, #2:int]]):
                Will move the vibrational branch from its current
                position relative to atom #1 to a simular relative
                position to atom #2. #1 and #2 refers to the indices
                of atoms 1 and 2.

        Returns:
            A dictionary that defines the anharmonic vibrational mode
        """
        # Either mode_number or mode_vector should be set

        if from_atom_to_atom is None:
            # currently this is required for there to be a translation
            raise NotImplementedError

        an_mode = {
            'type': 'translation',
        }

        # Adding the configurations to the mode object:

        # Prepares what is in the mode and calculated the tangent of
        # the mode at the relaxed structure.
        an_mode = get_trans_dict(from_atom_to_atom, self.vib)

        an_mode.update({
            # attach settings only for this mode
            'mode_settings': mode_settings
        })

        post_modes = self.get_post_modes()

        # Calculate the mode that should be removed from the harmonic
        # analysis
        mode_to_remove = calculate_highest_mode_overlap(
            an_mode['mode_tangent_mass_weighted'], post_modes)

        # Remove this mode from the normal mode spectrum.
        self.reduced_h_modes = np.delete(post_modes, mode_to_remove, axis=0)

        # TODO: ?? Should I Gramm-smidth orthogonalize?

        self.an_modes.append(an_mode)

        return an_mode

    def make_rotation_trajs(self):
        """Make trajectory files for the defined rotations

        These are helpful to check if the defined rotations corresponde
        to what was intended. Can be runned before the actual analysis.
        """
        for i, an_mode in enumerate(self.an_modes):
            if an_mode['type'] == 'rotation':
                AMA = RotAnalysis(an_mode,
                                  self.atoms,
                                  bak_filename='an_mode_'+str(i),
                                  traj_filename='an_mode_'+str(i)
                                  )
                AMA.make_rotation_traj(
                    30,
                    filename='rot_mode_'+str(i)+'.traj')

    def run(self):
        """Run the analysis"""
        for i, an_mode in enumerate(self.an_modes):
            if i > 0:
                warnings.warn(
                    "Warning: Module has not been tested for multiple \
                    anharmonic modes! Might not work properly.")

            if an_mode['type'] == 'rotation':
                AMA = RotAnalysis(
                    an_mode,
                    self.atoms,
                    bak_filename='an_mode_'+str(i),
                    traj_filename='an_mode_'+str(i),
                    settings=self.settings)

            elif an_mode['type'] == 'vibration':
                AMA = VibAnalysis(
                    an_mode,
                    self.atoms,
                    bak_filename='an_mode_'+str(i),
                    traj_filename='an_mode_'+str(i),
                    settings=self.settings)

            elif an_mode['type'] == 'translation':
                AMA = TransAnalysis(
                    an_mode,
                    self.atoms,
                    bak_filename='an_mode_'+str(i),
                    traj_filename='an_mode_'+str(i),
                    settings=self.settings)

            else:
                raise ValueError('unknown type')

            # adding ZPE, Z_mode, and energy_levels to mode object
            self.an_modes[i] = AMA.run()

        # Calculate the thermodynamical quantities:
        self.calculate_anharmonic_thermo()

    def calculate_anharmonic_thermo(self):
        """Calculates the thermodynamic quantities for the
        anharmonic normal mode analysis.
        """
        #
        # Calculate rotational mode ZPE and entropy energy
        #
        self.modes = self.get_post_modes()

        hnu_h_post = self.calculate_post_h_freqs()

        Z_all = 1.  # Global partition function
        ZPE = 0.  # zero point energy

        # The partition harmonic modes' partition functions
        for e in hnu_h_post:
            if e.imag == 0 and e.real >= 0.010:
                Z_mode = 1./(1.-np.exp(-e.real/(self.kT)))
                Z_all *= Z_mode

        # Adding zero-point energy of harmonic modes
        ZPE += self.get_ZPE_of_harmonic_subspace()

        an_energies = []
        for i, an_mode in enumerate(self.an_modes):
            # Add zero-point energy of mode
            ZPE += an_mode['ZPE']

            # Partition function of mode
            Z_mode = 0.
            for ei in an_mode['energy_levels']:
                Z_mode += np.exp(-(ei-an_mode['ZPE'])/self.kT)
            Z_all *= Z_mode

            # Print principle energy:
            # Difference between ZPE and first excited energy level
            e = an_mode['energy_levels'][1]-an_mode['energy_levels'][0]
            # Energy in inverse cm

            an_energies.append(e)

        self.hnu_h_post = hnu_h_post
        self.ZPE = ZPE
        self.Z_all = Z_all
        self.an_energies = an_energies
        self.entropic_energy = -1*self.kT*np.log(self.Z_all)

    def get_ZPE(self):
        return self.ZPE

    def get_entropic_energy(self):
        return self.entropic_energy

    def summary(self, log=None):
        """Summary of the vibrational frequencies.

        This is similar to the vibrational module, but it adds information
        about the anharmonic modes and prints the entropic energy
        contribution.

        Args:
            log : if specified, write output to a different location than
                stdout. Can be an object with a write() method or the name of
                a file to create.
        """

        if log is None:
            log = self.log

        if isinstance(log, str):
            log = paropen(log, 'a')
        write = log.write

        write(21*'-'+'\n')
        write('  #    meV     cm^-1    type'+'\n')
        for i, an_mode in enumerate(self.an_modes):
            e = self.an_energies[i]
            f = e*self.ev__inv_cm
            write('%3d %6.1f   %7.1f    %s \n' %
                  (i, 1000 * e, f, an_mode['type']))

        # Summaries the harmonic modes
        write(21*'-'+'\n')
        for i, e in enumerate(self.hnu_h_post):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ' '
                e = e.real
            f = e * self.ev__inv_cm  # frequency in cm^-1
            write('%3d %6.1f%s  %7.1f%s   %s \n' %
                  (i+len(self.an_modes), 1000 * e, c, f, c, 'Harmonic'))
        write(21*'-'+'\n')

        write('Zero-point energy: %.3f \n' % self.ZPE)
        write('Entropic energy: %.3f \n' % self.entropic_energy)

    def get_ZPE_of_harmonic_subspace(self, freq=False):
        """Zero-point energy of harmonic subspace
        Args:
            freq (optional[bool]): if we want frequencies in cm^-1

        Returns:
            Zero point energy
        """
        ZPE = 0.5 * self.hnu_h_post.real.sum()  # in meV
        if freq:
            ZPE *= self.ev__inv_cm
        return ZPE

    def get_post_modes(self):
        """Calculating the harmonic modes after orthogonalization with the
        newly defined modes from the anharmonic analysis modes.

        All modes here are in massweighted coordinates.

        We stack together the masscoordinated modes of the defined defined
        anharmonic mode objects. We then remove the weakest modes of the
        vibrational mode spectrum, and gramm smidth orthogonalize to the rest
        to the defined modes. We return the output.

        Returns:
            The harmonic subspace in mass weighted coordinates

        """
        if len(self.an_modes) > 0:
            for i, an_mode in enumerate(self.an_modes):
                an_mode_tangent = an_mode['mode_tangent_mass_weighted']
                if i == 0:
                    an_mode_tangents = an_mode_tangent
                else:
                    an_mode_tangents = np.vstack((an_mode_tangents,
                                                  an_mode_tangent))
            modes = gramm(np.vstack((an_mode_tangents, self.reduced_h_modes)))
        else:
            modes = self.reduced_h_modes

        return modes

    def calculate_post_h_freqs(self):
        """Calculate the frequencies of the harmonic subspace.

        This function has the side effect that it updates modes_post.

        Returns:
            Numpy array with the eigen energies of the harmonic modes
                in eV.
        """
        H = self.vib.H
        F = self.vib.im[:, None]*H*self.vib.im

        V_ = self.reduced_h_modes

        G = np.dot(V_, np.dot(F, V_.T))

        omega2_, modes_ = np.linalg.eigh(G)

        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        self.hnu_h_post = s * omega2_.astype(complex)**0.5  # Energies in eV

        self.modes_post = modes_

        return self.hnu_h_post

    def clean(self):
        for i, an_mode in enumerate(self.an_modes):
            os.remove('an_mode_'+str(i)+'.traj')
            os.remove('an_mode_'+str(i)+'.pckl')


def gramm(X):
    """Gramm-Schmidt orthogonalization

    Orthogonalization starts from the first vector in the matrix

    Args:
        X (2D numpy-array): mode vectors to be orthogonalized

    Returns:
        Numpy array of Gramm-Schmidt orthogonalized vector space
    """
    V = copy(X)
    n = len(X[0])
    for j in range(1, n):
        V[j] = V[j] - sum([
            np.inner(V[j], V[p])*V[p]/np.inner(V[p], V[p])
            for p in range(j)])
        V[j] = V[j]/(np.inner(V[j], V[j]))**(0.5)
    return np.array(V)


def calculate_highest_mode_overlap(tangent, modes):
    """Finding best projection mode:
    Calculates the projection of each mode on the tangent for the
    defined modes and returns the index of the mode that has the
    largest absolute projection on the mode tangent.
    """
    return np.argmin([np.abs(np.dot(tangent, mode)) for mode in modes])

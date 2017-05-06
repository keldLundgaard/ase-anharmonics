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
import warnings

from copy import copy

import numpy as np

import ase.units as units
from ase.parallel import paropen
from ase.io.trajectory import Trajectory

from define_rot_mode import get_rot_dict
from define_trans_mode import get_trans_dict
from anh_rot import RotAnalysis
from anh_vib import VibAnalysis
from anh_trans import TransAnalysis


class AnharmonicModes:
    """Calculate anharmonic modes: vibrational and rotational.

    The anharmonic mode extends the harmonic approximation of the
        vibrations module in ASE.

    See examples for how to use the module in the folder examples.

    """
    def __init__(
        self,
        vibrations_object,
        settings={},
        log=sys.stdout,
        pre_names='an_mode_',
    ):
        """Initialization

        Args:
            vibrations_object (obj): The vibration object of ASE.
            settings (optional[dict]): Change default behavior. E.g.
                "tempurature": 400  # (400K instead of 300K).
            log (optional[str]): name of log file.
        """
        self.vib = vibrations_object
        self.pre_names = pre_names
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
            orthogonalize=True,
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

        # Removing mode from harmonic spectrum
        self.reduced_h_modes = np.delete(
            self.reduced_h_modes,
            calculate_highest_mode_overlap(
                an_mode['mode_tangent_mass_weighted'],
                self.reduced_h_modes,
                print_out=self.settings.get('overlap_print', 0), ),
            axis=0)

        if orthogonalize:
            # Orthogonalize to the defined mode
            self.reduced_h_modes = gramm(np.vstack([
                an_mode['mode_tangent_mass_weighted'],
                self.reduced_h_modes]))[1:, :]

        self.check_defined_mode_overlap()

        return an_mode

    def define_translation(
            self,
            from_atom_to_atom=None,
            relax_axis=None,
            orthogonalize=True,
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

        # Prepares what is in the mode and calculated the tangent of
        # the mode at the relaxed structure.
        an_mode = get_trans_dict(from_atom_to_atom, self.vib)

        an_mode.update({'mode_settings': mode_settings})

        # If there should be an optimization along a direction
        an_mode.update({'relax_axis': relax_axis})

        # Calculate the mode that should be removed from the harmonic
        # analysis, and remove this mode from the normal mode spectrum.
        self.reduced_h_modes = np.delete(
            self.reduced_h_modes,
            calculate_highest_mode_overlap(
                an_mode['mode_tangent_mass_weighted'],
                self.reduced_h_modes,
                print_out=self.settings.get('overlap_print', 0), ),
            axis=0)

        if orthogonalize:
            # Orthogonalize to the defined mode
            self.reduced_h_modes = gramm(np.vstack([
                an_mode['mode_tangent_mass_weighted'],
                self.reduced_h_modes]))[1:, :]

        self.an_modes.append(an_mode)

        self.check_defined_mode_overlap()

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
        an_mode = {'type': 'vibration', }

        if mode_vector is not None:
            # This functionality should be easy to implement.
            # I have left it here to illustrate that it could be added,
            # but it will not be implemented before a usage shows up.
            raise NotImplementedError

        elif mode_number is not None:
            an_mode.update({
                # mode tangent that defines the vibrational mode fully
                'mode_tangent_mass_weighted': (
                    self.reduced_h_modes[mode_number]),
                # The mode that we have selected
                'mode': self.reduced_h_modes[mode_number],
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
            self.reduced_h_modes = np.delete(
                self.reduced_h_modes, mode_number, axis=0)
        else:
            raise NotImplementedError(
                'Need input of either mode_number of mode_vector ')

        self.an_modes.append(an_mode)

        self.check_defined_mode_overlap()

        return an_mode

    def check_defined_mode_overlap(self):
        """Checks if there are overlap between the last defined mode
        and the previous defined modes. Raises a warning if the overlap
        is higher than a set tolerance.
        """
        tol = self.settings.get('defined_modes_overlap_tol', 1e-6)
        if len(self.an_modes) > 1:
            for i in range(len(self.an_modes)-1):
                projection = np.abs(np.dot(
                    self.an_modes[i]['mode_tangent_mass_weighted'],
                    self.an_modes[-1]['mode_tangent_mass_weighted']))
                msg = ' '.join([
                    "Overlap between defined mode",
                    str(i), 'and', str(len(self.an_modes)-1),
                    'by', "%0.2e" % projection, '> tol:', "%0.2e" % tol])
                if projection > tol:
                    warnings.warn(msg)

    def make_rotation_trajs(self):
        """Make trajectory files for the defined rotations

        These are helpful to check if the defined rotations corresponde
        to what was intended. Can be runned before the actual analysis.
        """
        for i, an_mode in enumerate(self.an_modes):
            if an_mode['type'] == 'rotation':
                AMA = RotAnalysis(an_mode,
                                  self.atoms,
                                  an_filename='an_mode_'+str(i),
                                  )
                AMA.make_rotation_traj(
                    30,
                    filename='rot_mode_'+str(i)+'.traj')

    def run(self):
        """Run the analysis.

        Adding ZPE, Z_mode, and energy_levels to mode object.
        """
        for i, _ in enumerate(self.an_modes):
            AMA = self.get_analysis_object(i)
            self.an_modes[i] = AMA.run()

        self.calculate_anharmonic_thermo()

    def inspect_anmodes(self):
        """Run the analysis"""
        for i, an_mode in enumerate(self.an_modes):
            AMA = self.get_analysis_object(i)

            AMA.make_inspection_traj()

    def get_analysis_object(self, i):
        """Return the mode object for index i.
        """
        an_mode = self.an_modes[i]

        if an_mode['type'] == 'rotation':
                AMA = RotAnalysis(
                    an_mode,
                    self.atoms,
                    an_filename=self.pre_names+str(i),
                    settings=self.settings)

        elif an_mode['type'] == 'vibration':
            AMA = VibAnalysis(
                an_mode,
                self.atoms,
                an_filename=self.pre_names+str(i),
                settings=self.settings)

        elif an_mode['type'] == 'translation':
            AMA = TransAnalysis(
                an_mode,
                self.atoms,
                an_filename=self.pre_names+str(i),
                settings=self.settings)
        else:
            raise ValueError('unknown type')

        return AMA

    def calculate_anharmonic_thermo(self):
        """Calculates the thermodynamic quantities for the
        anharmonic normal mode analysis.
        """
        self.hnu_h_post = self.calculate_post_h_freqs()

        # the partition function for each mode
        self.ZPE_modes = []  # Zero point energy for mode
        self.Z_modes = []  # partition function for mode
        self.e_exitation_modes = []  # principle energy of mode

        #entropy contribution per vibrational mode
        entropic_energy_modes = []

        for i, an_mode in enumerate(self.an_modes):
            # Partition function of mode
            Z_mode = 0.
            dZ_dT = 0.
            for ei in an_mode['energy_levels']:
                Z_mode += np.exp(-(ei-an_mode['ZPE'])/self.kT)
                dZ_dT += (
                    np.exp(-(ei - an_mode['ZPE']) / self.kT) *
                    (ei - an_mode['ZPE']))

            entropic_energy = self.kT * np.log(Z_mode) + 1./Z_mode * dZ_dT
            entropic_energy_modes.append(entropic_energy)

            # Difference between ZPE and first excited energy level
            e_min_exitation = (
                an_mode['energy_levels'][1] - an_mode['energy_levels'][0])

            self.ZPE_modes.append(an_mode['ZPE'])
            self.Z_modes.append(Z_mode)
            self.e_exitation_modes.append(e_min_exitation)

        # The partition harmonic modes' partition functions
        for e in self.hnu_h_post:
            if e.imag == 0 and e.real >= 0.010:
                Z_mode = 1./(1.-np.exp(-e.real/(self.kT)))
                ZPE = 0.5 * e.real
                e_min_exitation = e.real
                x = e_min_exitation/self.kT
                entropic_energy_modes.append(
                    self.kT * (x/(np.exp(x) - 1.) - np.log(1. - np.exp(-x))))
            else:
                ZPE = 0.
                Z_mode = 1.
                e_min_exitation = 0.
                entropic_energy_modes.append(0.)

            self.ZPE_modes.append(ZPE)
            self.Z_modes.append(Z_mode)
            self.e_exitation_modes.append(e_min_exitation)

        # Make class variable
        self.entropic_energy_modes = entropic_energy_modes

        # Overall information
        self.ZPE = np.sum(self.ZPE_modes)
        self.entropic_energy = np.sum(entropic_energy_modes)

    def get_ZPE(self):
        return self.ZPE

    def get_entropic_energy(self):
        return self.entropic_energy

    def get_harmonic_thermo(self, hnu):
        ZPE_hmodes = []  # Zero point energy for mode
        Z_hmodes = []  # partition function for mode
        e_excitation_hmodes = []  # principle energy of mode
        entropic_energy_hmodes = []  # entropy per harmonic mode

        for e in hnu:
            if e.imag == 0 and e.real >= 0.010:
                Z_mode = 1./(1. - np.exp(-e.real/(self.kT)))
                ZPE = 0.5 * e.real
                e_min_exitation = e.real
                x = e_min_exitation/self.kT
                entropic_energy_hmodes.append(
                    self.kT * (x / (np.exp(x) - 1.) - np.log(1. - np.exp(-x))))
            else:
                ZPE = 0.
                Z_mode = 1.
                e_min_exitation = 0.
                entropic_energy_hmodes.append(0.)
            ZPE_hmodes.append(ZPE)
            Z_hmodes.append(Z_mode)
            e_excitation_hmodes.append(e_min_exitation)

        entropic_energy = np.sum(entropic_energy_hmodes)
        ZPE_h = np.sum(ZPE_hmodes)

        return {
            'ZPE_hmodes': ZPE_hmodes,
            'Z_hmodes': Z_hmodes,
            'e_excitation_hmodes': e_excitation_hmodes,
            'entropic_energy_hmodes': entropic_energy_hmodes,
            'entropic_energy': entropic_energy,
            'ZPE_h': ZPE_h}

    def pre_summary(self, log=None):
        """Summary of the harmonic mode analysis. This makes it easier to
        see what the anharmonic analysis contributes with.

        Args:
            log : if specified, write output to a different location than
                stdout. Can be an object with a write() method or the name of
                a file to create.
        """

        hnu_thermo = self.get_harmonic_thermo(self.vib.hnu)

        if log is None:
            log = self.log

        if isinstance(log, str):
            log = paropen(log, 'a')
        write = log.write

        write('Harmonic mode analysis: \n')
        write(27*'-'+'\n')
        write('  #    ZPE    E_exc   E_ent '+'\n')
        write('  #    meV    meV     meV '+'\n')
        write(27*'-'+'\n')
        for i, (ZPE, E_ex, E_entry) in enumerate(
                zip(hnu_thermo['ZPE_hmodes'],
                    hnu_thermo['e_excitation_hmodes'],
                    hnu_thermo['entropic_energy_hmodes'])):
            write(
                '%3d %6.1f %6.1f %7.1f  \n' %
                (i, 1000 * ZPE, 1000 * E_ex, 1000 * E_entry))

        write(27*'-'+'\n')

        write('Zero-point energy: %.3f eV \n' % hnu_thermo['ZPE_h'])
        write('Entropic energy: %.3f eV \n' % hnu_thermo['entropic_energy'])
        write('\n')

    def summary(self, log=None):
        """Summary of the vibrational frequencies.

        This is similar to the vibrational module, but it adds information
        about the anharmonic modes and prints the entropic energy
        contribution.

        E_exc is the excitation energy going from ZPE to the first
        accessible vibrational energy.
        E_entropy is the entropy contribution of the mode.

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

        write('An-harmonic analysis \n')
        write(40*'-'+'\n')
        write('  #    ZPE    E_exc   E_ent  type'+'\n')
        write('  #    meV    meV     meV '+'\n')
        write(40*'-'+'\n')
        for i, an_mode in enumerate(self.an_modes):
            write(
                '%3d %6.1f %6.1f %7.1f    %s \n' %
                (i, 1000 * self.ZPE_modes[i],
                    1000 * self.e_exitation_modes[i],
                    1000 * self.entropic_energy_modes[i],
                    an_mode['type']))
        write(40*'*'+'\n')
        for i in range(len(self.an_modes), len(self.ZPE_modes)):
            write(
                '%3d %6.1f %6.1f %7.1f    %s \n' %
                (i, 1000 * self.ZPE_modes[i],
                    1000 * self.e_exitation_modes[i],
                    1000 * self.entropic_energy_modes[i],
                    'Harmonic'))
        write(40*'-'+'\n')

        write('Zero-point energy: %.3f eV \n' % self.ZPE)
        write('Entropic energy: %.3f eV \n' % self.entropic_energy)

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

        # If we still have harmonic modes
        if len(G) > 0:
            omega2_, modes_ = np.linalg.eigh(G)

            s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
            self.hnu_h_post = s * omega2_.astype(complex)**0.5  # in eV

            self.modes_post = modes_
        else:
            # There are no harmonic modes
            self.hnu_h_post = []
            self.modes_post = []

        return self.hnu_h_post

    def clean(self):
        """Clean up directory for all files related to anharmonic
        analysis.

        Checking if the first part is equal to self.pre_names
        E.g. if self.pre_names='an_mode_': an_mode_1.traj is deleted
        """
        for fn in os.listdir(os.getcwd()):
            if len(fn.split(self.pre_names)[0]) == 0:
                os.remove(fn)

    def write_h_spacemodes(self, n=None, kT=units.kB * 300, nimages=30):
        """Write mode number n to trajectory file. If n is not specified,
        writes all non-zero modes."""
        if n is None:
            for index, energy in enumerate(self.hnu_h_post):
                if abs(energy) > 1e-5:
                    self.write_h_spacemodes(n=index, kT=kT, nimages=nimages)
            return

        mode = self.reduced_h_modes[n] * np.sqrt(kT / abs(self.hnu_h_post[n]))
        p = self.atoms.positions.copy()
        traj = Trajectory('%sHarmonic_%02d.traj' % (self.pre_names, n), 'w')
        calc = self.atoms.get_calculator()
        self.atoms.set_calculator()

        for x in np.linspace(0, 2 * np.pi, nimages, endpoint=False):
            pos_delta = np.zeros_like(p)
            pos_delta[self.vib.indices] += (
                np.sin(x) * mode.reshape((len(self.vib.indices), 3)))
            self.atoms.set_positions(p + pos_delta)
            traj.write(self.atoms)
        self.atoms.set_positions(p)
        self.atoms.set_calculator(calc)
        traj.close()


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


def calculate_highest_mode_overlap(tangent, modes, print_out=False):
    """Finding best projection mode:
    Calculates the projection of each mode on the tangent for the
    defined modes and returns the index of the mode that has the
    largest absolute projection on the mode tangent.

    Returns:
        Index of mode (int)
    """
    projections = [np.abs(np.dot(tangent, mode)) for mode in modes]
    if print_out:
        print('Projections of anharmonic mode on harmonic modes:')
        string = ""
        for i, p in enumerate(projections):
            if i and i % 5 == 0:
                string += "\n"
            string += "%3d: %.2f  " % (i, p**2)
        print(string + '\n')

    return np.argmax(projections)

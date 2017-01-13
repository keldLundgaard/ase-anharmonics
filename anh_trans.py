import sys

# import numpy as np
# from ase.io.trajectory import Trajectory

from anh_base import BaseAnalysis
# from fit_rots import PeriodicFit
# from fit_settings import fit_settings


class TransAnalysis(BaseAnalysis):
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
        super(TransAnalysis, self).__init__()

        self.an_mode = an_mode
        self.atoms = atoms
        self.traj_filename = traj_filename
        self.bak_filename = bak_filename
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
        self.rel_Z_mode_change_tol = settings.get('rel_Z_mode_tol', 0.005)

        self.initialize()

    def initial_sampling(self):
        """ Function to start initial sampling of the rotational
        mode. This can be done before extra samples are introduced.
        """

        # initializing
        if len(self.an_mode.get('translations', [])) == 0:
            self.an_mode['translations'] = self.get_initial_points()
            # self.add_rot_energy(None)  # adding ground state

        # getting initial data points
        raise

        # while (len(self.an_mode['rot_angles']) >
        #        len(self.an_mode.get('rot_energies', []))):

        #     next_angle = self.an_mode['rot_angles'][
        #         len(self.an_mode['rot_energies'])]

        #     self.add_rot_energy(next_angle)

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

            # Fit mode
            raise
            fit_settings.update({
                'symnumber': self.an_mode['symnumber'],
                'verbose': False,
                'search_method': 'iterative',
            })

            fitobj = PeriodicFit(fit_settings)

            if self.fit_forces:
                fitobj.set_data(
                    self.an_mode['translations'],
                    self.an_mode['trans_energies'],
                    self.an_mode.get('trans_forces', []))
            else:
                fitobj.set_data(
                    self.an_mode['translations'],
                    self.an_mode['trans_energies'],
                    [])

            fitobj.run()

            ZPE, Z_mode, energies = self.get_thermo(fitobj)

            self.ZPE_hist.append(ZPE)
            self.Z_mode_hist.append(Z_mode)

        return ZPE, Z_mode, energies

    def get_initial_points(self, nsamples=5):
        raise

    # def sample_new_point(self):

    # def add_rot_energy(self, angle):

    # def get_initial_angles(self, nsamples=5):
    #     """ Returns at which initial angles the energy calculations
    #     should be done.
    #     0 and 2pi is not necessary, as these are already included. """
    #     angles = (
    #         2.*np.pi /
    #         ((nsamples+1)*self.an_mode['symnumber'])
    #         * np.array(range(0, nsamples+1)))
    #     return angles

    # def get_rotate_positions(self, angle):
    #     """ Get the atomic positions of branch that are rotated
    #     after the branch has been rotated by angle.

    #     Args:
    #         angle (float): angle of rotation in radians
    #     returns:
    #         rot_pos (numpy array): the positions of the atoms in the
    #             branch.
    #     """

    #     rot_pos = copy(self.groundstate_positions)

    #     rot_pos[self.an_mode['branch']] = rotatepoints(
    #         self.an_mode['base_pos'],
    #         self.an_mode['rot_axis'],
    #         angle,
    #         rot_pos[self.an_mode['branch']])

    #     return rot_pos

    # def make_rotation_traj(self, num_angles, filename='inspect_an_mode.traj'):
    #     """ make a rotational traj file to easily inspect the defined
    #     rotational mode.
    #     """

    #     traj = Trajectory(filename, mode='w', atoms=self.atoms)

    #     old_pos = self.atoms.positions.copy()
    #     calc = self.atoms.get_calculator()
    #     self.atoms.set_calculator()

    #     angles = self.get_initial_angles(nsamples=num_angles)

    #     for angle in angles:
    #         new_pos = self.get_rotate_positions(angle)
    #         self.atoms.set_positions(new_pos)
    #         traj.write(self.atoms)
    #         self.atoms.set_positions(old_pos)

    #     self.atoms.set_calculator(calc)
    #     traj.close()

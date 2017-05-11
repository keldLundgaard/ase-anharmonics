import sys

from ase.build import molecule, fcc111, add_adsorbate
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

sys.path.append("../..")

from __init__ import AnharmonicModes

slab = fcc111('Al', size=(2, 2, 2), vacuum=3.0)

CH3 = molecule('CH3')
add_adsorbate(slab, CH3, 3.0, 'ontop')

constraint = FixAtoms(mask=[a.symbol == 'Al' for a in slab])
slab.set_constraint(constraint)

slab.set_calculator(EMT())

dyn = QuasiNewton(slab, trajectory='relax.traj')
dyn.run(fmax=0.01)

vib = Vibrations(slab, indices=[8, 9, 10, 11])
vib.run()
vib.summary()
vib.write_mode()

print('\n >> Anharmonics <<\n')

AM = AnharmonicModes(
    vibrations_object=vib,
    settings={
        # 'use_forces': True,
        # 'plot_mode': True,
        'temperature': 298.15,
        # 'rot_nsamples': 15,
        # 'max_step_iterations': 1,
        # 'min_step_iterations': 1,
        'plot_mode_each_iteration': True,
        # 'rot_nsamples': 9,
        # 'fit_verbose': True,
        'fit_plot_regu_curve_iterations': True,
        'fit_plot_regu_curve': True,
        'fit_forces': False,
        'verbosity': 2,
        'rel_Z_mode_tol': 0.05,
        # 'fit_max_order': 5,
        # 'fit_pdiff': 2,
    })

# rot_mode = AM.define_rotation(
#     basepos=[0., 0., -1.],
#     branch=[9, 10, 11],
#     symnumber=3)

AM.define_vibration(mode_number=5)

AM.clean()
AM.run()
# AM.inspect_anmodes()
# AM.pre_summary()
AM.summary()
# AM.write_h_spacemodes()

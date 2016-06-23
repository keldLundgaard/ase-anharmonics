import sys
import os

from ase.structure import molecule
from ase.lattice.surface import fcc111, add_adsorbate
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations
import ase.io

sys.path.append("../..")

from __init__ import AnharmonicModes

if os.path.exists('CH3_Al.traj'):
    slab = ase.io.read('CH3_Al.traj')
    slab.set_calculator(EMT())  # Need to reset when loading from traj file
else:
    slab = fcc111('Al', size=(2, 2, 2), vacuum=3.0)

    CH3 = molecule('CH3')
    add_adsorbate(slab, CH3, 2.5, 'ontop')

    constraint = FixAtoms(mask=[a.symbol == 'Al' for a in slab])
    slab.set_constraint(constraint)

    slab.set_calculator(EMT())

    dyn = QuasiNewton(slab, trajectory='QN_slab.traj')
    dyn.run(fmax=0.05)

    ase.io.write('CH3_Al.traj', slab)

# Running vibrational analysis
vib = Vibrations(slab, indices=[8, 9, 10, 11])
vib.run()
vib.summary()

print('\n >> Anharmonics <<\n')
AM = AnharmonicModes(vibrations_object=vib)
vib_mode = AM.define_vibration(mode_number=-1)
AM.run()
AM.summary()

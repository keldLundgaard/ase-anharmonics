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
add_adsorbate(slab, CH3, 2.5, 'ontop')

constraint = FixAtoms(mask=[a.symbol == 'Al' for a in slab])
slab.set_constraint(constraint)

slab.set_calculator(EMT())

dyn = QuasiNewton(slab)
dyn.run(fmax=0.05)

# Running vibrational analysis
vib = Vibrations(slab, indices=[8, 9, 10, 11])
vib.run()
vib.summary()

AM = AnharmonicModes(vibrations_object=vib)
vib_mode = AM.define_vibration(mode_number=-1)

AM.inspect_anmodes()  # creates trajectory file
AM.run()
AM.pre_summary()
AM.summary()

# Delete all the generated files
# vib.clean()
# AM.clean()

import sys

from ase.build import molecule, fcc111, add_adsorbate
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

sys.path.append("../..")

from __init__ import AnharmonicModes

slab = fcc111('Au', size=(2, 2, 2), vacuum=4.0)
H = molecule('H')
add_adsorbate(slab, H, 3.0, 'bridge')

constraint = FixAtoms(mask=[a.symbol == 'Au' for a in slab])
slab.set_constraint(constraint)

slab.set_calculator(EMT())

dyn = QuasiNewton(slab)
dyn.run(fmax=0.05)

# Running vibrational analysis

vib = Vibrations(slab, indices=[8])
vib.run()
vib.summary()
print()

AM = AnharmonicModes(
    vibrations_object=vib,
    pre_names='an_mode_relax_',
    settings={
        'plot_mode': True,
    })

# Translation by moving from top position on 4 to 6
AM.define_translation(
    from_atom_to_atom=[4, 6],
    relax_axis=[0, 0, 1])
AM.inspect_anmodes()  # creates trajectory file
AM.run()
AM.pre_summary()
AM.summary()

# Delete all the generated files
# vib.clean()
# AM.clean()

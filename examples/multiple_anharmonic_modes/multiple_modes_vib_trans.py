import sys

from ase.structure import molecule
from ase.lattice.surface import fcc111, add_adsorbate
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

sys.path.append("../..")

from __init__ import AnharmonicModes

slab = fcc111('Au', size=(2, 2, 2), vacuum=4.0)
H = molecule('H')
add_adsorbate(slab, H, 3.0, 'ontop')

constraint = FixAtoms(mask=[a.symbol == 'Au' for a in slab])
slab.set_constraint(constraint)

slab.set_calculator(EMT())

QuasiNewton(slab).run(fmax=0.05)

vib = Vibrations(slab, indices=[8])
vib.run()
vib.summary()
vib.clean()

AM = AnharmonicModes(
    vibrations_object=vib,
    settings={'plot_mode': True})

AM.define_vibration(mode_number=-1)
AM.define_translation(from_atom_to_atom=[4, 6])

AM.inspect_anmodes()
AM.run()
AM.summary()
AM.clean()

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

if 0 and os.path.exists('H_Au.traj'):
    slab = ase.io.read('H_Au.traj')
    slab.set_calculator(EMT())  # Need to reset when loading from traj file
else:
    slab = fcc111('Au', size=(2, 2, 2), vacuum=4.0)

    H = molecule('H')
    add_adsorbate(slab, H, 3.0, 'ontop')

    constraint = FixAtoms(mask=[a.symbol == 'Au' for a in slab])
    slab.set_constraint(constraint)

    slab.set_calculator(EMT())

    dyn = QuasiNewton(slab, trajectory='QN_slab.traj')
    dyn.run(fmax=0.05)

    ase.io.write('H_Au.traj', slab)

# Running vibrational analysis
vib = Vibrations(slab, indices=[8])
vib.run()
vib.summary()
vib.write_mode()
# vib.clean()
print('\n >> Anharmonics <<\n')

AM = AnharmonicModes(
    vibrations_object=vib,
    settings={'plot_mode': True})

translational_mode = AM.define_translation(
    from_atom_to_atom=[4, 6]  # move from top position on 4 to 6
)

AM.inspect_anmodes()  # creates trajectory file
AM.run()
AM.summary()

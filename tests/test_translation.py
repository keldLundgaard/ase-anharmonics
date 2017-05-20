import sys
sys.path.append("..")

from ase.build import molecule, fcc111, add_adsorbate
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

from __init__ import AnharmonicModes

slab = fcc111('Au', size=(2, 2, 2), vacuum=4.0)
H = molecule('H')
add_adsorbate(slab, H, 3.0, 'ontop')

constraint = FixAtoms(mask=[a.symbol == 'Au' for a in slab])
slab.set_constraint(constraint)
slab.set_calculator(EMT())

dyn = QuasiNewton(slab, logfile='/dev/null')
dyn.run(fmax=0.05)

vib = Vibrations(slab, indices=[8])
vib.run()
vib.summary(log='/dev/null')
vib.clean()

AM = AnharmonicModes(vibrations_object=vib)

translational_mode = AM.define_translation(
    from_atom_to_atom=[4, 6]  # move from top position on 4 to 6
)
AM.run()
AM.summary(log='/dev/null')
AM.clean()

# print(AM.get_ZPE(), AM.get_entropic_energy())
assert abs(AM.get_ZPE() - 0.168) < 1e-3, AM.get_ZPE()
assert abs(AM.get_entropic_energy() - (0.0293)) < 1e-3, (
    AM.get_entropic_energy())

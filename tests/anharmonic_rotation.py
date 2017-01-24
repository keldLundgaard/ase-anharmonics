import sys
sys.path.append("..")

from ase.build import molecule, fcc111, add_adsorbate
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

from __init__ import AnharmonicModes

slab = fcc111('Al', size=(2, 2, 2), vacuum=3.0)
CH3 = molecule('CH3')
add_adsorbate(slab, CH3, 2.5, 'ontop')

constraint = FixAtoms(mask=[a.symbol == 'Al' for a in slab])
slab.set_constraint(constraint)
slab.set_calculator(EMT())

dyn = QuasiNewton(slab, logfile='/dev/null')
dyn.run(fmax=0.05)

vib = Vibrations(slab, indices=[8, 9, 10, 11])
vib.run()
vib.summary(log='/dev/null')
vib.clean()

AM = AnharmonicModes(vibrations_object=vib)
rot_mode = AM.define_rotation(
    basepos=[0., 0., -1.],
    branch=[9, 10, 11],
    symnumber=3)

AM.run()
# AM.summary(log='/dev/null')
AM.summary()
AM.clean()

assert abs(AM.get_ZPE() - 0.277) < 1e-3, AM.get_ZPE()
assert abs(AM.get_entropic_energy() - (-0.068)) < 1e-3, (
    AM.get_entropic_energy())

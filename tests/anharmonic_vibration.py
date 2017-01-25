from __future__ import print_function
import sys
sys.path.append("..")

from ase.build import molecule
from ase.optimize import QuasiNewton
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

from __init__ import AnharmonicModes

H2 = molecule('H2')
H2.set_calculator(EMT())
dyn = QuasiNewton(H2, logfile='/dev/null')
dyn.run(fmax=0.05)

vib = Vibrations(H2, indices=[0, 1])
vib.run()
vib.summary(log='/dev/null')
vib.clean()

AM = AnharmonicModes(vib,
                     settings={
                         'temperature': 1000,
                     })
vib_mode = AM.define_vibration(mode_number=-1)
AM.run()
AM.summary(log='/dev/null')
AM.clean()

assert abs(AM.get_ZPE() - 0.497) < 1e-3, AM.get_ZPE()
assert abs(AM.get_entropic_energy()) < 1e-3, AM.get_entropic_energy()

AM = AnharmonicModes(vib,
                     settings={
                         'temperature': 1000,
                         'step_multi_kT_disp': 1.0,
                     })

vib_mode = AM.define_vibration(mode_number=-1, mode_settings={})

AM.run()
AM.summary(log='/dev/null')
AM.clean()

assert abs(AM.get_ZPE() - 0.379) < 1e-3, AM.get_ZPE()
assert abs(AM.get_entropic_energy()) < 1e-3, AM.get_entropic_energy()

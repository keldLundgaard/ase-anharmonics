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
dyn = QuasiNewton(H2)
dyn.run(fmax=0.05)

vib = Vibrations(H2, indices=[0, 1])
vib.run()
vib.summary()
vib.clean()

AM = AnharmonicModes(vib,
                     settings={
                         'temperature': 1000,
                         'max_disp': 1.,
                     })
vib_mode = AM.define_vibration(mode_number=-1)
AM.run()
AM.summary()
AM.clean()

assert abs(AM.get_ZPE() - 1.231) < 1e-3
assert abs(AM.get_entropic_energy()) < 1e-3

print('\n >> Anharmonics 1000k with initial estimated out to 1.0 kT<<\n')

AM = AnharmonicModes(vib,
                     settings={
                         'temperature': 1000,
                         'step_multi_kT_disp': 1.0,
                         'max_disp': 1.,
                         # 'plot_energy_surface': 1  # display plot
                     })

vib_mode = AM.define_vibration(mode_number=-1, mode_settings={})

AM.run()
AM.summary()
AM.clean()

assert abs(AM.get_ZPE() - 0.400) < 1e-3
assert abs(AM.get_entropic_energy()) < 1e-3

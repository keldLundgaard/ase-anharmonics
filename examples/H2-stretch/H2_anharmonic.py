import sys
sys.path.append("../..")
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

print('\n >> Anharmonics 1000k sampling up to 0.25 kT<<\n')

AM = AnharmonicModes(vib,
                     settings={
                         'temperature': 1000,
                         'max_disp': 1.,
                         'plot_mode': True
                     })

vib_mode = AM.define_vibration(mode_number=-1)

AM.run()
AM.inspect_anmodes()
AM.summary()

print('\n >> Anharmonics 1000k with initial estimated out to 1.0 kT<<\n')

AM = AnharmonicModes(vib,
                     pre_names='an_mode_1000K_',
                     settings={
                         'temperature': 1000,
                         'step_multi_kT_disp': 1.0,
                         'max_disp': 1.,
                         'plot_mode': True
                     })

vib_mode = AM.define_vibration(mode_number=-1, mode_settings={})
AM.inspect_anmodes()
AM.run()
AM.summary()

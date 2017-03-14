import numpy as np
import sys
import time

sys.path.append("..")

from energy_spectrum_solver import energy_spectrum

# Square well
if 0:
    xmin = 0.
    xmax = 1.
    fval = lambda x: 0.  # return zero -- infinite square well
    Hcoeff = 1.

    energies = energy_spectrum(
        xmin, xmax, fval, Hcoeff,
        Romberg_integrator=True,
        mode='fast',  # 'accurate'
    )

    analytical = np.pi**2/(2.*(xmax-xmin)**2)*np.arange(1, len(energies)+1)**2

    numprint = 50
    diff = energies[:numprint]-analytical[:numprint]
    scaled_diff = np.abs(diff)*np.exp(-0.015*analytical[:numprint])
    print(energies[:numprint])
    print(analytical[:numprint])
    print(diff[:numprint])
    print(scaled_diff[:numprint])
    loss = np.sum(scaled_diff)
    print(loss)
    # return loss

# Harmonic oscillator
if 0:
    xmin = -50
    xmax = 50

    # Harmonic potential
    fval = lambda x: 0.5*x**2
    Hcoeff = 1.

    energies = energy_spectrum(
        xmin, xmax, fval, Hcoeff,
        Romberg_integrator=True,
        mode='fast',
    )

    analytical = 0.5+np.arange(0, len(energies))

    numprint = 50

    diff = energies[:numprint]-analytical[:numprint]
    print(energies)
    print(analytical)
    print(diff)

    # scaled_diff = 1e9*np.abs(diff)*np.exp(-0.5*analytical[:numprint])
    # loss = np.sum(scaled_diff)
    # return loss


def squarewell_benchmark(
        mode, gridincrements, neigbors,
        minimalgrid=None, Romberg=True):
    fval = lambda x: 0.  # return zero -- infinite square well

    energies = energy_spectrum(
        0., 1., fval, 1.,
        Romberg_integrator=Romberg,
        neighbors=neigbors,
        mode=mode,
        minimalgrid=minimalgrid,
        gridincrements=gridincrements,
        incrementfactor=None,
        verbose=0)
    numprint = 20
    analytical = np.pi**2/(2.*1.**2)*np.arange(1, len(energies)+1)**2
    diff = energies[:numprint]-analytical[:numprint]
    scaled_diff = np.abs(diff)*np.exp(-0.015*analytical[:numprint])
    loss = np.sum(scaled_diff)
    return loss


def harmonic_benchmark(
        mode, gridincrements, neigbors,
        minimalgrid=None, Romberg=True):
    # Harmonic potential
    fval = lambda x: 0.5*x**2

    energies = energy_spectrum(
        -50, 50, fval, 1.,
        Romberg_integrator=Romberg,
        neighbors=neigbors,
        mode=mode,
        minimalgrid=minimalgrid,
        gridincrements=gridincrements,
        incrementfactor=None,
        verbose=0)
    numprint = 20
    analytical = 0.5+np.arange(0, len(energies))
    diff = energies[:numprint]-analytical[:numprint]
    scaled_diff = 1e9*np.abs(diff)*np.exp(-0.5*analytical[:numprint])
    loss = np.sum(scaled_diff)
    return loss


def run_test(mode, gridincrements, neigbors, Romberg, minimalgrid):
    t0 = time.time()
    loss = np.square(
        harmonic_benchmark(
            mode, gridincrements, neigbors,
            minimalgrid=minimalgrid, Romberg=Romberg)
        * squarewell_benchmark(
            mode, gridincrements, neigbors,
            minimalgrid=minimalgrid, Romberg=Romberg))
    log_loss = np.log(loss)
    t1 = time.time()
    timing = t1-t0
    print('Romberg', Romberg)
    print(timing, 'loss', log_loss)


# mode = 1
# gridincrements = 3
# neigbors = 6
# Romberg = False
# minimalgrid = None
# run_test(mode, gridincrements, neigbors, Romberg, minimalgrid)

# mode = 1
# gridincrements = 2
# neigbors = 6
# Romberg = True
# minimalgrid = None
# run_test(mode, gridincrements, neigbors, Romberg, minimalgrid)
print('double')
mode = 1
gridincrements = 5
neigbors = 6
Romberg = True
minimalgrid = 128
run_test(mode, gridincrements, neigbors, Romberg, minimalgrid)

print('ref')
mode = 1
gridincrements = 0
neigbors = 6
Romberg = False
minimalgrid = 1024
run_test(mode, gridincrements, neigbors, Romberg, minimalgrid)

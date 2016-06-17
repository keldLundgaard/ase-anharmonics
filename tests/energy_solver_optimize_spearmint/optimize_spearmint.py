import numpy as np
import sys
sys.path.append("../..")

from energy_spectrum_solver import energy_spectrum


def squarewell_benchmark(mode, gridincrements, neigbors):
    fval = lambda x: 0.  # return zero -- infinite square well

    energies = energy_spectrum(
        0., 1., fval, 1.,
        Romberg_integrator=True,
        neighbors=neigbors,
        mode=mode,
        minimalgrid=None,
        gridincrements=gridincrements,
        incrementfactor=None,
        verbose=0)

    numprint = 20
    analytical = np.pi**2/(2.*1.**2)*np.arange(1, len(energies)+1)**2
    diff = energies[:numprint]-analytical[:numprint]
    scaled_diff = np.abs(diff)*np.exp(-0.015*analytical[:numprint])
    loss = np.sum(scaled_diff)
    return loss


def harmonic_benchmark(mode, gridincrements, neigbors):
    # Harmonic potential
    fval = lambda x: 0.5*x**2

    energies = energy_spectrum(
        -50, 50, fval, 1.,
        Romberg_integrator=True,
        neighbors=neigbors,
        mode=mode,
        minimalgrid=None,
        gridincrements=gridincrements,
        incrementfactor=None,
        verbose=0)

    numprint = 20
    analytical = 0.5+np.arange(0, len(energies))
    diff = energies[:numprint]-analytical[:numprint]
    scaled_diff = 1e6*np.abs(diff)*np.exp(-0.5*analytical[:numprint])
    loss = np.sum(scaled_diff)
    return loss


# Write a function like this called 'main'
def main(job_id, params):
    print('Run #%d' % job_id)
    print(params)

    mode = params['mode'][0]
    gridincrements = params['gridincrements'][0]
    neigbors = params['neigbors'][0]

    loss = np.log(np.square(
        harmonic_benchmark(mode, gridincrements, neigbors)
        * squarewell_benchmark(mode, gridincrements, neigbors)))

    return loss

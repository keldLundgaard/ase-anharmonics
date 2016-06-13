"""Library for solving the 1d schrodinger equation using finite Distance
and the Romberg integration correction.
"""

# TODO: Current implementation does not use Romberg integration.
# Should be corrected.
#

import numpy as np


lapjj = [
    [0],
    [-2.0, 1.0],
    [-5./2, 4./3, -1./12],
    [-49./18, 3./2, -3./20, 1./90],
    [-205./72, 8./5, -1./5, 8./315, -1./560],
    [-5269./1800, 5./3, -5./21, 5./126, -5./1008, 1./3150],
    [-5369./1800, 12./7, -15./56, 10./189, -1./112, 2./1925, -1./16632]]

lapbli = [
    [0],
    [-2.0, 1.0],
    [-5./2, 4./3, -1./12],
    [-490./180, 270./180, -27./180, 2./180],
    [-14350./5040, 8064./5040, -1008./5040, 128./5040, -9./5040],
    [-73766./25200, 42000./25200, -6000./25200, 1000./25200, -125./25200,
        8./25200],
    [-2480478./831600, 1425600./831600, -222750./831600, 44000./831600,
        -7425./831600, 864./831600, -50./831600],
    [-228812298./75675600, 132432300./75675600, -22072050./75675600,
        4904900./75675600, -1003275./75675600, 160524./75675600,
        -17150./75675600, 900./75675600]]

boundarycorr = [
    [0],
    [0],
    [10./12, -15./12, -4./12, 14./12, -6./12, 1./12],
    [126./180, -70./180, -486./180, 855./180, -670./180, 324./180, -90./180,
        11./180],
    [3044./5040, 2135./5040, -28944./5040, 57288./5040, -65128./5040,
        51786./5040, -28560./5040, 10424./5040, -2268./5040, 223./5040],
    [13420./25200, 29513./25200, -234100./25200, 540150./25200,
        -804200./25200, 888510./25200, -731976./25200, 444100./25200,
        -192900./25200, 56825./25200, -10180./25200, 838./25200],
    [397020./831600, 1545544./831600, -11009160./831600, 29331060./831600,
        -53967100./831600, 76285935./831600, -83567088./831600,
        70858920./831600, -46112220./831600, 22619850./831600,
        -8099080./831600, 1999044./831600, -304260./831600, 21535./831600],
    [32808524./75675600, 188699914./75675600, -1325978220./75675600,
        4020699410./75675600, -8806563220./75675600, 15162089943./75675600,
        -20721128428./75675600, 22561929390./75675600, -19559645820./75675600,
        13424150740./75675600, -7206307108./75675600, 2963338014./75675600,
        -901775420./75675600, 191429035./75675600, -25318020./75675600,
        1571266./75675600]]


def energy_spectrum(
        xmin, xmax,
        fval, Hcoeff,
        minimalgrid=200,
        neighbors=2,
        Romberg_integrator=True,
        gridincrements=3):

    """Calculate energy spectrum

    Note:
        This module does not use the Romberg integrator as I think there
        is a problem with it.

        incrementfactor = 2
        eigenarray = np.zeros((gridincrements+1, minimalgrid), float)
        pointarray = np.array(
            (minimalgrid+1)
            * incrementfactor**np.arange(0, gridincrements+1)-0.5, int)

        realincrementfactors = np.zeros(len(pointarray))
        # check that these are identical, otherwise convergence is poor:
        for i in range(1, len(pointarray)):
            realincrementfactors[i] = (pointarray[i]+1.0)/(pointarray[i-1]+1.0)

        for i in range(gridincrements+1):
            eigenarray[i] = FDsolver(
                xmin, xmax, pointarray[i], fval, Hcoeff,
                correction=False,
                neighbors=2)[:minimalgrid]

        extrapolatedspectrum, relativeerrors = RombergSpectrumIntegrator(
            eigenarray, realincrementfactors)

        print eigenarray[0]
        print eigenarray[gridincrements]
        print extrapolatedspectrum

        energy_spectrum = extrapolatedspectrum

    Args:
        xmin (float): lower end of domain
        xmax(float): upper end of domain
        n (int): number of points
        fval (object): function for 'RHS'
        Hcoeff ():coefficient to multiply the FD matrix with
        correction -- boundary correction for FD matrix
        neighbours -- order of FD solver
            (higher convergence for more neighbours)
    Returns:
        eigenvalue spectrum (numpy array)
    """
    energy_spectrum = FDsolver(
        xmin, xmax, minimalgrid*4, fval, Hcoeff,
        correction=False,
        neighbors=2)[:minimalgrid]

    return energy_spectrum


def FDsolver(
        xmin, xmax, n, fval, Hcoeff,
        correction=False,
        neighbors=2):
    """

    Builds an FD matrix for the equation system
    u''(x) = f(x)*u(x)

    Sets up the structure of H as in terms of the standard FD stencil
    of -1/2*LAPLACIAN

    Solves this using the sparse scipy solver or with numpy's eigenvalue solver

    Args:
        xmin (float): lower end of domain
        xmax(float): upper end of domain
        n (int): number of points
        fval (object): function for 'RHS'
        Hcoeff ():coefficient to multiply the FD matrix with
        correction -- boundary correction for FD matrix
        neighbours -- order of FD solver
            (higher convergence for more neighbours)

    Returns:
        eigenvalue spectrum (numpy array)
    """
    # Insure tha the bounds are properly defined
    assert xmax > xmin

    # Distance between grid points
    h = (xmax-xmin)/(n - 1.0)

    # The grid array and source term
    potential = np.zeros(n)
    x0 = np.zeros(n)
    for i in range(n):
        x0[i] = xmin + i*h
        potential[i] = fval(x0[i])

    # Initialization of Main Matrix
    H = np.zeros((n, n))
    for i, c in enumerate(lapbli[neighbors]):
        H.flat[n * i::n + 1] = -0.5 * c / h**2
        H.flat[i:n*(n-i)+1:n + 1] = -0.5 * c / h**2

    # Setting boundary correction of H using modified
    # FD stencil of -1/2*LAPLACIAN of the same order
    if correction:
        corrstencil = np.array(boundarycorr[neighbors])
        lencorr = len(corrstencil)

        # Fixing the first and last row first
        H.flat[0:lencorr-1] = -0.5 * corrstencil[1:] / h**2
        H.flat[-(lencorr-1):] = -0.5 * (corrstencil[1:])[::-1]/h**2

        # All the in between rows
        for i in range(1, neighbors-1):
            H.flat[n*i:n*i+lencorr] = -0.5*corrstencil / h**2
            H.flat[-n*i-lencorr:-n*i] = -0.5*corrstencil[::-1] / h**2
            corrstencil = np.insert(corrstencil, [0], 0.)
            lencorr = len(corrstencil)
    H *= Hcoeff

    # Adding the potential
    H += np.diag(potential)

    if not correction:
        # H symmetric, eigenvalues are real
        eigenvaluesrough = np.linalg.eigvalsh(H)
        eigenvalues = np.sort(eigenvaluesrough)
    else:
        # H not symmetric -> eigenvalues could have imaginary component,
        # careful here..
        eigenvaluesrough = np.linalg.eigvals(H)
        eigenvalues = np.sort(np.abs(eigenvaluesrough))

    return eigenvalues


def RombergSpectrumIntegrator(spectrum, realincrementfactors):
    """
    Args:
        Spectrum ():
        realincrementfactors ():

    """
    # Number of points used to solve the initial equation
    n = len(spectrum[0, :])

    extrapolatedspectrum = np.zeros(n)
    relativeerrors = np.zeros(n)
    for i in range(n):
        rombergeigenvals, relerr, convexps, best, besterr = \
            RombergIntegrator(spectrum[:, i], realincrementfactors[0])
        extrapolatedspectrum[i] = best
        relativeerrors[i] = besterr
    return extrapolatedspectrum, relativeerrors


def RombergIntegrator(integrants, realincrementfactor=2, exact=None):
    """"""
    n = len(integrants)

    extrapolants = np.zeros((n, n))
    convexps = np.zeros((n, n))
    relativeerrors = np.zeros((n, n))

    extrapolants[0, :] = integrants
    for i in range(1, n):
        extr, convexp = \
            RichardsonExtrapolator(
                extrapolants[i-1, i-1:], realincrementfactor, order=2*i)
    extrapolants[i, i-1:] = extr

    if exact is None:
        bestextrap = extrapolants[n-1, n-1]
    else:
        bestextrap = exact

    for i in range(n):
        for j in range(i+1, n):
            pre = bestextrap-extrapolants[i, j-1]
            aft = bestextrap-extrapolants[i, j]
            convexps[i, j] = np.log(pre/aft)/np.log(realincrementfactor)

    for i in range(n):
        for j in range(i, n):
            # print np.abs(1-extrapolants[i, j])
            # print bestextrap
            relativeerrors[i, j] = (
                int(np.log(np.abs(1-extrapolants[i, j]/(bestextrap+1e-24)))
                    / np.log(10)))

    bestdiaelement = 0
    i = 1
    while i < n and abs(convexps[i-1, i]-2.0*i) < 2.0:
        i += 1
    bestdiaelement = i-1

    bestextrapolantvalue = extrapolants[bestdiaelement, bestdiaelement]

    tmp1 = (
        extrapolants[bestdiaelement-1, bestdiaelement-1]
        - extrapolants[bestdiaelement-1, bestdiaelement])
    tmp2 = extrapolants[bestdiaelement-1, bestdiaelement]+1e-24
    errestimate = int(np.log(np.abs(tmp1/tmp2+1e-24))/np.log(10))
    return (
        extrapolants, relativeerrors, convexps,
        bestextrapolantvalue, errestimate)


def RichardsonExtrapolator(approxarr, factor, order=2):
    """"""
    length = len(approxarr)
    extrapolant = np.zeros(length)
    exponents = np.zeros(length)
    for r in range(1, length):
        extrapolant[r] = (
            approxarr[r]+(approxarr[r]-approxarr[r-1])/(factor**order-1))
        if r > 2:
            exponents[r] = ConvergenceExponent(
                extrapolant[r-2], extrapolant[r-1], extrapolant[r], factor)
    return extrapolant, exponents


def ConvergenceExponent(new, newer, newest, increment):
    """"""
    exponent = np.log((new-newer)/(newer-newest))/np.log(increment)
    return exponent

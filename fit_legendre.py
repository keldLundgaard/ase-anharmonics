import numpy as np

from fit_base import BaseFit
# import basislib


class NonPeriodicFit(BaseFit):
    """
    Introduction:
    -------------
    Fit for a non periodic function using Legendra basis

    Attributes:
    -----------
    basis_coeff_matrix    -- Save a matrix with precalculated coefficients
                             for cheaper function evaluations
    """

    def __init__(self, settings):
        if settings['verbose']:
            print('Initialize Non-Periodic fit...'),
        super(NonPeriodicFit, self).__init__(settings)
        self.settings['basistype'] = 'legendra'
        self.basisfunction = self.basefunc
        self.basis_coeff_matrix = None
        if settings['verbose']:
            print('[DONE]')

    def basefunc(self, xval, ndiff):
        """ Calls the polynomial basis function.
        """

        # Get differentiated x polynomial vector
        # E.g. 0 diff => x = [1, x^1, x^2, x^3, x^4...]
        xpoly = getxpolyvec(xval, ndiff, self.order)

        # Legendre polynomial matrix multiplied with x
        basis_vec = np.dot(self.basis_coeff_matrix.T, xpoly)

        return basis_vec

    def getgamma(self, order):
        """ Smoothness operator for Legendra basis functions
        The first 1's (equal to smoothing function order) are put to zero
        """
        gamma = np.eye(order)
        p = self.settings['pdiff']
        gamma[0:p, 0:p] = 0
        return gamma

    def setpersistentbasis(self):
        """ Calculating basefunc coefficients is costly,
        do this only once for all evaluations """
        if self.settings['verbose']:
            print('calculating Legendre basis...'),
        self.basis_coeff_matrix = Legendra_intbasis(
            self.order, self.settings['pdiff'])
        if self.settings['verbose']:
            print('[DONE]]'),


def getxpolyvec(xval, ndiff, order):
    """
    Returns [1, x^1, x^2, x^3, x^4...] differentiated ndiff times
    E.g. 1 diff => [0, 1, 2x, 3x^2, 4x^3...]
    """
    xpoly = np.zeros(order)
    # Adding values if not differentiating more times than highest x order
    if ndiff < order:
        # x = [1, x^1, x^2, x^3, x^4...]
        xpoly[ndiff:] = np.array([xval**n for n in range(order-ndiff)])
        # The extra coefficient from differentiating ndiff times
        if ndiff > 0:
            # Dirty quick factorial
            diffcoeff = 1
            for i in range(2, ndiff + 1):
                diffcoeff *= i

            # Calculate the next coefficient by using the previous factorial
            xpoly[ndiff] *= diffcoeff
            for i in range(ndiff + 1, order):
                diffcoeff *= i
                diffcoeff /= (i - ndiff + 1)
                xpoly[i] *= diffcoeff
    return xpoly


def Legendra_intbasis(order, xint, xdiffnormalize=True):
    """
    Construct the Lengendra Polynomials on matrix form,
    integrate the polynomials xint times (to give a twice differeated
        smoothing function)
    Finally normalize them so that the smoothing function is equal to identity
    """

    # Get the Legendra polynomials in matrix form
    Tint = Legendra_basis(order)

    # Apply recursive integration formula xint times
    for i in range(xint):
        Tint = Legendra_matrix_integrator(Tint, order)

    # Normalization so that differenting the function twice
    # and multiplying basis with themselves will yields
    # an identity matrix (smoothness function)
    if xdiffnormalize:
        for i in range(xint, order):
            for j in range(order):
                Tint[i, j] = Tint[i, j] * np.sqrt((2. * (i - xint) + 1.) / 2.)
    return Tint


def Legendra_matrix_integrator(T, order):
    """
    Legendre integrated matrix, recurence formula
    Indices are shifted so instad of
        1/(2n+1)*(P_{n+1}-P{n-1})
    this function uses
        1/(2n+1)*(P_{n}-P{n-2})
    This is the same as putting the first row (index 0)
      as row index -1 in formula1
    """
    Tint = np.zeros((order, order))

    assert(order > 1), "Must have more than one data point"

    # Integrate first two Polynomial functions
    Tint[0, 0] = 1. / (2. * 0 - 1.) * T[0, 0]
    Tint[1, 1] = 1. / (2. * 1 - 1.) * T[1, 1]

    for i in range(2, order):
        for j in range(order):
            Tint[i, j] = 1. / (2. * i - 1.) * (T[i, j] - T[i - 2, j])
    return Tint


def Legendra_basis(order):
    """
    Matrix with the coefficients to the Legendra polynomials.
    Polynomials given from the recursion formulae:
      P_n = (2n-1)/(n)*x*P_{n-1} - (n-1)/n*P_n-2

    For an order of 5 the function will output matrix on following form:
           1,     x,     x2,    x3,    x4
    =====================================
    [
     [     1,     0,      0,     0,     0],
     [     0,     1,      0,     0,     0],
     [ -1/2.,     0,   3/2.,     0,     0],
     [     0, -3/2.,      0,  5/2.,     0],
     [  3/8.,     0, -30/8.,     0, 35/8.]
    ]
    The P's are then given by
    P = M*x where x = [1, x^1, x^2, x^3, x^4]
    """
    assert(order > 1), "Must have more than one data point"
    T = np.zeros((order, order))
    T[0, 0] = 1
    T[1, 1] = 1
    for i in range(2, order):
        for j in range(order-1):
            T[i, j + 1] += (2. * i - 1.)/i * T[i - 1, j]
            T[i, j] += -1. * (i - 1.) / i * T[i - 2, j]
    return T

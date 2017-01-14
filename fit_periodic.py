import numpy as np

from fit_base import BaseFit
# import basislib


class PeriodicFit(BaseFit):
    """
    Introduction:
    -------------
    Fit for a periodic function.
    Cos and Sin are used as basis functions.

    Attributes:
    -----------
    basis_matrix    --
    """

    def __init__(self, settings):
        if settings['verbose']:
            print('Initialize Periodic fit...'),

        super(PeriodicFit, self).__init__(settings)
        self.settings['basistype'] = 'trigonometric'
        self.basisfunction = self.periodicbasefunc

        if settings['verbose']:
            print('[DONE]')

    def periodicbasefunc(self, angle, ndiff):
        """ Calls the trigonometric basis function.
        self.order is set later """
        return self.trigXrow(
            angle,
            ndiff,
            self.settings['pdiff'],
            self.settings['symnumber'],
            self.order)

    def getgamma(self, order):
        """ Smoothness operator for periodic basis functions """
        gamma = np.eye(order)
        # This value should be checked
        gamma[0, 0] = 0
        return gamma

    def trigXrow(self, theta, ndiff, pdiff, symnumber, order):
        """ Calculates each trigonometric basis function values for

        F = a_0 + sum_{n=1}^{(order-1)/2}( a_n/(n*s)**(pdiff) cos (n*O*s)
                                         + b_n/(n*s)**(pdiff) sin (n*O*s))
        n = function order
        O = angle of rotation (running variable)
        s = symmetry number
        pdiff = pdiff penalty (smoothness factor to avoid overfitting)

        Assume base wave length is 4*pi

        Order must be uneven to have cos and sin to each order
        In T first element correspont to a_0
        Then follows T[if i % 2 == 1] = a_{(i+ i%2)/2}+b_{i/2}


        Args:
            theta (float): Input angle
            ndiff (int): Number of derivates on basis
            pdiff (int): A specific smoothing order (corresponding
                to the overall fitting function)
            symnumber (int): The symmetry number
            order (int): Number of basis functions

        Returns:
            Xrow (numpy array): Row for the design matrix for the
                fitting problem.

        """

        Xrow = np.zeros(order)

        # First basis value given no derivate
        if ndiff == 0:
            Xrow[0] = 1.

        # k-coefficient power
        coeffdiff = ndiff-pdiff

        # The basis functions change with nth derivation
        basefuncs = [None, None]

        if ndiff % 4 == 0:
            basefuncs = [self.cos_pos, self.sin_pos]
        elif ndiff % 4 == 1:
            basefuncs = [self.sin_neg, self.cos_pos]
        elif ndiff % 4 == 2:
            basefuncs = [self.cos_neg, self.sin_neg]
        else:
            basefuncs = [self.sin_pos, self.cos_neg]

        # Range excludes last basis value if order is even
        # which must be added afterwards
        for nthbase in range(2, order, 2):
            # k = sym*2pi/lambda
            # lambda = 2*2pi/n
            # k = sym*n/2
            k = symnumber*nthbase/2
            coeff = k**coeffdiff
            Xrow[nthbase-1] = coeff*basefuncs[0](k*theta)
            Xrow[nthbase] = coeff*basefuncs[1](k*theta)

        # Adding a last cos/sin function, half base
        #   if even number of orders
        if order % 2 == 0:
            k = symnumber*(order/2)
            coeff = k**coeffratio
            Xrow[order-1] = coeff*basefuncs[0](k*theta)
        return Xrow

    def cos_pos(self, angle):
        return np.cos(angle)

    def sin_pos(self, angle):
        return np.sin(angle)

    def cos_neg(self, angle):
        return -np.cos(angle)

    def sin_neg(self, angle):
        return -np.sin(angle)

    def setpersistentbasis(self):
        pass

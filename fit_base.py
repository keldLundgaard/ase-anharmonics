import abc

import numpy as np

from fit_funcs import RR, find_optimal_regularization


class BaseFit:
    """Base class for fitting module

    Calculates the bestfit function to a set of data points, function values
    (and optionally function derivate values) that the user provides.
    The module extends into periodic function fit or polynomial fit
    depending on whether the function is expected to be repeating or not.

    The module creates a best fit by regularization:
        Cost = (Xa-y)**2 + omega2*(Gamma*(a-p))**2


    The fit assumes that user sets regular data values at first n positions
        and if derivatives are present when setting the fitting data the
        the extra derivative info is put into the next n positions

    Function values are evaluated using fval after fit

    Attributes:
    -----------
    settings        -- All external tweakable settings
    coeffs          -- Vector with best fit ceofficients for current basis
    xvals           -- Original input coordinates for fitting (sorted)
    yvals           -- The measured function values at input coordinates
    yders           -- The derivates of function values at input coordinates
    basisfunction   -- The actual function to calculate a basis
    order           -- The number of coefficients used when fitting
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, settings):
        self.settings = settings
        self.coeffs = []
        self.cleardata()
        self.basisfunction = self.undefinedbasisfunc
        self.order = -1

    def set_data(self, xvals, yvals, yderivates=[]):
        self.xvals = xvals
        self.yvals = yvals
        self.yders = yderivates

        # check data
        self.use_derivatives = False
        assert len(xvals) == len(yvals)
        if len(yderivates) > 0:
            assert len(xvals) == len(yderivates)
            self.use_derivatives = True

    def cleardata(self):
        """ Clear working array for fitting"""
        self.xvals = []
        self.yvals = []
        self.yders = []

    def undefinedbasisfunc(x, ndiff):
        """ Returns the value of x in a basis at nth differential
        from the basic basis function """
        raise

    def xvalsinbasis(self, xvals, use_derivatives=False):
        """ Creating the design matrix

        Args:
            xvals (numpy array): Sampled points along the given mode

            yders (numpy array): Derivatives to the energy along the mode

        Returns:
            X (numpy array): Design matrix for the fitting problem
        """

        self.setpersistentbasis()

        # The standard design matrix for y vals and y' vals
        if use_derivatives:
            X = np.zeros((2*len(xvals), self.order))
        else:
            X = np.zeros((len(xvals), self.order))

        # Use the basisfunctions without any derivate
        for row, x in enumerate(xvals):
            X[row, :] = self.basisfunction(x, 0)

        # Extend X matrix to include derivative of f at x
        if use_derivatives:
            for row, x in enumerate(xvals):
                X[row+len(xvals), :] = self.basisfunction(x, 1)

        return X

    def run(self):
        """Creating a best fit by regularization:

        The cost function is given as follows:
        Cost = (Xa-y)**2 + omega2*(Gamma*(a-p))**2

        Use the transformation
        a_ = Gamma*a
        p_ = Gamma*p
        X_g = Gamma^(-1)*X
        y_ = y
        To get
        Cost = (X*a-y)**2 + omega2*((a_-p_))**2
        a_ = Min(Cost, a_)
        a = Gamma^(-1)*a_

        """

        # require more than 3 points
        assert len(self.xvals) > 3

        # Sets number of fitting coefficients to be used
        #
        if self.use_derivatives:
            # self.order = self.getorder(len(self.xvals)+len(self.yders))
            self.order = self.get_order(len(self.xvals))
        else:
            self.order = self.get_order(len(self.xvals))

        min_order = 3
        max_order = self.get_order(len(self.xvals))

        EPE_min_list = []

        res_list = []

        order_list = np.arange(min_order, max_order+1, 2)

        for order in order_list:
            self.order = order
            # Setting up ydata
            y = self.scale_measureddata(self.yvals, self.yders)

            # Setting up design matrix
            X = self.xvalsinbasis(
                self.xvals,
                use_derivatives=self.use_derivatives)

            # constant prior
            p = np.zeros(self.order)
            p[0] = np.min(self.yvals)

            # Finding the optimal omega2 (regularzation parameter)

            res = find_optimal_regularization(X, y, p)
            opt_omega2, omega2_list, epe_list, err_list, ERR_list = res

            res_list.append(res)

            EPE_min = np.min(epe_list)
            EPE_min_list.append(EPE_min)

        best_arg = np.argmin(np.array(EPE_min_list))

        res = res_list[best_arg]

        opt_omega2, omega2_list, epe_list, err_list, ERR_list = res

        # Finding the optimal solution
        a0, neff = RR(X, y, p, opt_omega2)

        # Printing the optimal coeffs
        if self.settings.get('verbose'):
            print('The optimal coefficients:')
            print(', '.join(["{0:0.4f}".format(i) for i in a0]))
            print("Effective number of parameters : %.2f" % neff)
            print("Estimated Prediction Error : %.2f" % EPE_min)
            print("Optimal Regularization strength : %.3e" % opt_omega2)

        self.coeffs = a0
        self.n_eff = neff
        self.opt_omega2 = opt_omega2
        self.omega2_list = omega2_list
        self.epe_list = epe_list
        self.err_list = err_list
        self.ERR_list = ERR_list

    def scale_basismatrix(self, basismatrix):
        """ Scaling the row corresponding to derivatives """
        weight = self.settings['derivateive_weight']
        rowstart = len(self.xvals)
        basismatrix[rowstart:, :] = np.multiply(
            basismatrix[rowstart:, :], weight)
        return basismatrix

    def scale_measureddata(self, yvals, yders):
        """ Scaling the derivatives of the function
        to have more or less impact on the fitting procedure
        """
        weight = self.settings['derivateive_weight']
        yders_scaled = np.multiply(yders, weight)
        return np.concatenate((yvals, yders_scaled))

    def fval(self, x):
        """ Returns the vaue of the optimal function at a point
        The function is used after executing the fitting procedure
        The function accepts both lists and scalars
        """
        # If scalar or list
        diff_order = 0  # Want the predicted value, not its derivate
        if not hasattr(x, '__len__'):
            b = self.basisval(x, diff_order)
            y = np.dot(b, self.coeffs)
        else:
            y = np.zeros(len(x))
            for i, xi in enumerate(x):
                b = self.basisval(xi, diff_order)
                y[i] = np.dot(b, self.coeffs)
        return y

    def basisval(self, x, ndiff):
        """ Returns the vector of x in current basis """
        return self.basisfunction(x, ndiff)

    def get_order(self, N):
        """ Returns the number of coefficients used for fitting function.
        N - is the number of training data points
        """
        if N <= 3:
            order = N
        else:
            # Always an uneven number of basis functions
            order = N - N % 2 - 1

        order = min(order, self.settings.get('max_order', 1000))
        return order

    @abc.abstractmethod
    def setpersistentbasis(self):
        """ Setting the basis vector transformation"""
        pass

    def plot_regularization_curve(self, name_add=''):
        # Matplotlib is loaded selectively as it is requires
        # libraries that are often not installed on clusters
        import matplotlib.pylab as plt

        an_name = self.settings.get('an_name', 'xx')
        filename = an_name + '_regu_curve' + name_add + '.png'

        sort_order = np.argsort(self.omega2_list)
        omega2_arr = np.array(self.omega2_list).take(sort_order)
        epe_arr = np.array(self.epe_list).take(sort_order)
        err_arr = np.array(self.err_list).take(sort_order)
        ERR_arr = np.array(self.ERR_list).take(sort_order)

        plt.title('omega2: %.2e Neff: %.1f' % (self.opt_omega2, self.n_eff))

        plt.loglog(omega2_arr, epe_arr, '-', label='EPE')
        plt.loglog(omega2_arr, np.sqrt(err_arr), '--', label='err')
        plt.loglog(omega2_arr, np.sqrt(ERR_arr), '-.', label='ERR')

        plt.legend(loc=2)
        plt.xlabel('Omega2')
        plt.ylabel('eV')
        # plt.show()
        plt.savefig(filename)
        plt.clf()

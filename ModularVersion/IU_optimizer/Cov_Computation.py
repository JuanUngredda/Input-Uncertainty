
import numpy as np
from scipy.linalg import lapack, blas
from scipy import linalg
from scipy.linalg import lapack, blas

class COV_computation():

    def __init__(self, model):
        self.model = model
        self.X = model.X
        self.kern = model.kern

    def partial_precomputation_for_covariance(self):
        """
        Computes the posterior covariance between points.
        :param X1: some input observations
        :param X2: other input observations
        """
        self.woodbury_chol = self.model.posterior._woodbury_chol
        self.X = self.model.X

    def posterior_covariance_between_points_partially_precomputed(self, X1, X2):
        """
        Computes the posterior covariance between points.

        :param kern: GP kernel
        :param X: current input observations
        :param X1: some input observations
        :param X2: other input observations
        """

        Kx1 = self.kern.K(self.X, X1)
        Kx2 = self.kern.K(self.X, X2)
        K12 = self.kern.K(X1, X2)

        tmp1 = self.dtrtrs(self.woodbury_chol, Kx1)[0]
        tmp2 = self.dtrtrs(self.woodbury_chol, Kx2)[0]
        var = K12 - tmp1.T.dot(tmp2)

        return var

    def dtrtrs(self,A, B, lower=1, trans=0, unitdiag=0):
        """
        Wrapper for lapack dtrtrs function
        DTRTRS solves a triangular system of the form
            A * X = B  or  A**T * X = B,
        where A is a triangular matrix of order N, and B is an N-by-NRHS
        matrix.  A check is made to verify that A is nonsingular.
        :param A: Matrix A(triangular)
        :param B: Matrix B
        :param lower: is matrix lower (true) or upper (false)
        :returns: Solution to A * X = B or A**T * X = B
        """
        A = np.asfortranarray(A)
        # Note: B does not seem to need to be F ordered!
        return lapack.dtrtrs(A, B, lower=lower, trans=trans, unitdiag=unitdiag)


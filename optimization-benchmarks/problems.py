import numpy as np


# TODO: make it quadratic problem
class SimpleProblem:

    def __init__(self, x0):
        self.x0 = x0

    def func(self, x):
        return np.linalg.norm(x-self.x0)**2

    def grad(self, x):
        return 2*(x-self.x0)

    def hess(self, x):
        return 2*np.identity(x.shape[0])


class LpApproxProblem:

    def __init__(self, dim, p, x, subspace):
        pass

    def grad(self, x):
        pass

    def hess(self, x):
        pass

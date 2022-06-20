#!/usr/bin/env python3

import numpy as np

from solvers import GradientSolver, NewtonSolver
from problems import SimpleProblem

def main():
    problem = SimpleProblem(x0=np.array([-1.5,0.5,0.1]))

    x0 = np.array([0.,0.,0.])
    gslvr = GradientSolver(lambda x: problem.grad(x))
    res = gslvr.solve(x0)
    print('gradient solver:', res)
    print('value:', problem.func(res['x']))

    nslvr = NewtonSolver(
        lambda x: problema.grad(x),
        lambda x: problem.hess(x)
    )
    res2 = nslvr.solve(x0)
    print('newton solver:', res2)
    print('value:', problem.func(res2['x']))


if __name__ == "__main__":
    main()


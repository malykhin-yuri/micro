import numpy as np


class GradientSolver:
    def __init__(self, grad):
        self.grad = grad

    def solve(self, x0, max_steps=100, eps=0.1):
        x = x0
        h = 0.3
        steps = 0
        while True:
            gr = self.grad(x)
            x = x - h * gr
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            if eps is not None and np.linalg.norm(gr) <= eps:
                break
        return {'x': x, 'grad': gr, 'steps': steps}


class NewtonSolver:
    def __init__(self, grad, hess):
        self.grad = grad
        self.hess = hess

    def solve(self, x0, max_steps=100, eps=0.1):
        x = x0
        h = 1
        steps = 0
        while True:
            gr = self.grad(x)
            x = x - h * np.linalg.inv(self.hess(x)) @ gr
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            if eps is not None and np.linalg.norm(gr) <= eps:
                break
        return {'x': x, 'grad': gr, 'steps': steps}

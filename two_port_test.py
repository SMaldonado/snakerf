import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

f = np.logspace(1,6,1000)
w = srf.f2w(f)

Z1 = srf.L(1e-6, w)
Z2 = srf.C(1e-9, w)

bser = np.array([[[1, -Z],[0, 1]] for Z in Z1])
bshunt = np.array([[[1, 0],[-srf.Z2Y(Z), 1]] for Z in Z2])

# bcascade = 

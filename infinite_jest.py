import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf

fspace = 1e3
fstart = 1e5
fstop = 200e6
w = srf.f2w(np.arange(fstart, fstop, fspace))
ESR = 1

print(srf.C(0, w))
print(srf.C(0, w, ESR))

print(srf.C(inf, w))
print(srf.C(inf, w, ESR))

# print(Vdiv)

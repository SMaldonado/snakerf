import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

f = np.logspace(1,7,1000)
w = srf.f2w(f)

Z1 = srf.L(1e-6, w)
Z2 = srf.Zopen(w) # srf.C(1e-9, w) #

filt = srf.Two_Port.from_network(f, [Z1], [Z2])
V2 = filt.V_out(50*np.ones(len(f)), 50*np.ones(len(f)))

print(V2)

plt.plot(filt.fs, V2.real)
plt.plot(filt.fs, V2.imag)
plt.show()

# srf.plot_power_spectrum(plt.gca(), filt.fs, srf.mag(V2)**2 / 50)
# plt.show()

# #
# bser = np.array([[[1, -Z],[0, 1]] for Z in Z1])
# bshunt = np.array([[[1, 0],[-srf.Z2Y(Z), 1]] for Z in Z2])
#
# bcascade = bser@bshunt
# print(bcascade)

import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

f = np.logspace(5,9,1000)
w = srf.f2w(f)

Z1 = srf.L(1e-6, w)
Z2 = srf.C(1e-9, w) # srf.Zopen(w) #

# filt = srf.Two_Port.from_network(f, [Z1, srf.ser(Z1, Z2), Z1], [Z2, Z2])

RLGC = srf.RLGC_from_stripline(fs = f, Dk = 4.6, Df = 0, R_ins = inf, h = 0.020, w = 0.01)

filt = srf.Two_Port.from_tl(f, RLGC, 2)

# print(filt.Z_in())
V2 = filt.V_out(srf.R(50, w), srf.R(50, w))

plt.subplot(2,1,1)
plt.semilogx(filt.fs, srf.dBv(srf.mag(V2)))
plt.twinx()
plt.semilogx(filt.fs, np.rad2deg(srf.phase(V2)), ls = '--')

plt.subplot(2,1,2)
plt.semilogx(filt.fs, filt.Z_in(srf.R(50, w)).real)
plt.twinx()
plt.semilogx(filt.fs, filt.Z_in(srf.R(50, w)).imag, ls = '--')

plt.show()

# srf.plot_power_spectrum(plt.gca(), filt.fs, srf.mag(V2)**2 / 50)
# plt.show()

# #
# bser = np.array([[[1, -Z],[0, 1]] for Z in Z1])
# bshunt = np.array([[[1, 0],[-srf.Z2Y(Z), 1]] for Z in Z2])
#
# bcascade = bser@bshunt
# print(bcascade)

import math
from math import pi
import numpy as np
from numpy import log10, rad2deg, deg2rad
import matplotlib.pyplot as plt
from snakerf import R, L, C, Znetwork, dBv, phase

C1 = 2.75e-9
C2 = 4.7e-9
C3 = C1
ESR_C = 0#0.01

L1 = 12.5e-9
L2 = L1
ESR_L = 0 #3.4e-3
TOL_L = 0.05

CHPF = 330e-12

RL = 50

fspace = 10e3
fstart = 1e6
fstop = 50e6
w = 2*pi*np.arange(fstart, fstop, fspace)

fig = plt.figure(figsize=(10,5))

# double T
series = [R(50, w), L(L1, w, ESR = ESR_L), L(L2, w, ESR = ESR_L), C(CHPF, w, ESR = ESR_C)]
shunt  = [C(C1, w, ESR = ESR_C), C(C2, w, ESR = ESR_C), C(C3, w, ESR = ESR_C), R(50, w)]



# C1 = 6.045e-12
# C2 = 6.567e-9
# C3 = 1.305e-12
# C4 = 21.88e-9
# ESR_C = 0
#
# L1 = 5.698e-6
# L2 = 5.245e-9
# L3 = 26.4e-6
# L4 = 1.574e-9
# ESR_L = 0
#
# # Bessel
# series = \
#     [R(50, w) + C(C1, w, ESR = ESR_C) + L(L1, w, ESR = ESR_L), \
#     C(C3, w, ESR = ESR_C) + L(L3, w, ESR = ESR_L)]
# shunt = \
#     [par(C(C2, w, ESR = ESR_C), L(L2, w, ESR = ESR_L)), \
#     par(par(C(C4, w, ESR = ESR_C), L(L4, w, ESR = ESR_L)), R(50, w))]

v, z_shunt_eq = Znetwork(series, shunt)
vin = v[0,:]
Zin = z_shunt_eq[0,:]
vout = v[-1,:]

axl = plt.gca()
axl.semilogx(w/(2*pi), dBv(vin), label = 'vin')
axl.semilogx(w/(2*pi), dBv(vout), label = 'vout')
plt.axvline(27.12e6 - 162e3, c = 'black', ls = '--')
plt.axvline(27.12e6 + 162e3, c = 'black', ls = '--')
axl.grid()
axl.set_ylabel('mag (dB)')

plt.legend()

axr = axl.twinx()
# axr.semilogx(w/(2*pi), Zin.real, c = 'red')
# axr.semilogx(w/(2*pi), Zin.imag, c = 'green')
axr.semilogx(w/(2*pi), rad2deg(phase(vin)) , c = 'purple')
axr.semilogx(w/(2*pi), rad2deg(phase(vout)) , c = 'gold')
# axr.semilogx(w/(2*pi), rad2deg(phase(vout, unwrap = False)) , c = 'gold', ls = '--')
# axr.semilogx(w/(2*pi), rad2deg(phase(vin)), ls = '--', label = 'vin')
# axr.semilogx(w/(2*pi), rad2deg(phase(vout)), ls = '--', label = 'vout')
# axr.set_ylabel('phase (deg)')
# plt.xlim(27e6, 27.5e6)


plt.show()

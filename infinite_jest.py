import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi
from snakerf import R, L, C, Zopen, Znetwork, w2f, dBv

fspace = 1e3
fstart = 1e5
fstop = 200e6
w = srf.f2w(np.arange(fstart, fstop, fspace))

C1 = 2.75e-9
C2 = 4.7e-9
C3 = C1
ESR_C = 0#0.01

L1 = 12.5e-9
L3 = 2.5e-9
L2 = L1 - L3

ESR_L = 0 #3.4e-3
TOL_L = 0.05

CHPF = 330e-12

RL = 50


fig = plt.figure(figsize=(10,5))

R1 = R(50, w)
C1 = C(C1, w, ESR = ESR_C)
L1 = L(L1, w, ESR = ESR_L)
C2 = C(C2, w, ESR = ESR_C)
L2 = L(L2, w, ESR = ESR_L)
L3 = L(L3, w, ESR = 0)
C3 = C(C3, w, ESR = ESR_C)
CHPF = C(CHPF, w, ESR = ESR_C)
ZO = Zopen(w)
RL = R(50, w)
# double T
series = [R1, L1, L2, L3, CHPF]
shunt  = [C1, C2, ZO, C3, RL]

v, z_shunt_eq = Znetwork(series, shunt)
vin = v[0,:]
Zin = z_shunt_eq[0,:]
vout = v[-1,:]

axl = plt.gca()
axl.semilogx(w2f(w), dBv(vin), label = 'vin')
axl.semilogx(w2f(w), dBv(vout), label = 'vout')
# plt.axvline(27.12e6 - 162e3, c = 'black', ls = '--')
# plt.axvline(27.12e6 + 162e3, c = 'black', ls = '--')
axl.grid()
axl.set_ylabel('mag (dB)')
plt.gca().xaxis.set_major_formatter(srf.HzFormatter)

plt.show()
# print(Vdiv)

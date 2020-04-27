# this file's filename no longer makes sense

import snakerf as srf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from math import inf

fig = plt.figure(figsize=(10,5))

npts = 100000
fstart = 10e3
fstop = 2e9
# fspace = 10e3
w = srf.f2w(np.logspace(np.log10(fstart), np.log10(fstop), npts))
# w = srf.f2w(np.arange(fstart, fstop, fspace))

L = 2.2e-6
ESR = 0.25
SRF = 120e6
Qref = 10
fQref = 10e6
Zsrf = 1500

ZL1 = srf.L(L, w, ESR=ESR)
ZL2 = srf.L(L, w, ESR=ESR, Qref=Qref, fQref=fQref)
ZL3 = srf.L(L, w, ESR=ESR, srf=SRF)
ZL4 = srf.L(L, w, ESR=ESR, srf=SRF, Zsrf = Zsrf)
ZL5 = srf.L(L, w, ESR=ESR, Qref=Qref, fQref=fQref, srf=SRF, Zsrf = Zsrf)


plt.loglog(srf.w2f(w), srf.mag(ZL1), label = 'ESR')
plt.loglog(srf.w2f(w), srf.mag(ZL2), label = 'ESR, Qref')
plt.loglog(srf.w2f(w), srf.mag(ZL3), label = 'ESR, SRF')
plt.loglog(srf.w2f(w), srf.mag(ZL4), label = 'ESR, SRF, Zsrf')
plt.loglog(srf.w2f(w), srf.mag(ZL5), label = 'ESR, SRF, Zsrf, Qref')
plt.grid()
plt.legend()
plt.gca().xaxis.set_major_formatter(srf.HzFormatter)
plt.xlabel('frequency')
plt.ylabel(r'|Z| ($\Omega$)')
plt.title('inductor model impedances vs. frequency')
# plt.gca().twinx()
# plt.plot(srf.w2f(w)/1.0e6, srf.phase(ZL, deg = True), c = 'orange')
# plt.loglog(srf.w2f(w)/1.0e6, Q, c = 'purple')

plt.show()

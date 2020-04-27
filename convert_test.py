# this file's filename no longer makes sense

import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf

fig = plt.figure(figsize=(10,5))

fspace = 1e3
fstart = 1e5
fstop = 200e6
w = srf.f2w(np.arange(fstart, fstop, fspace))

L = 2.2e-6
ESR = 0.25
SRF = 120e6
Qref = inf
fQref = inf
Zsrf = 1500

ZL = srf.L(L, w, ESR=ESR, Qref=Qref, fQref=fQref, srf=SRF, Zsrf = Zsrf)

plt.loglog(srf.w2f(w)/1.0e6, srf.mag(ZL))
plt.grid()
plt.gca().twinx()
plt.plot(srf.w2f(w)/1.0e6, srf.phase(ZL, deg = True), c = 'orange')
# plt.loglog(srf.w2f(w)/1.0e6, Q, c = 'purple')

plt.show()

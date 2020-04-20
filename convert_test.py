# this file's filename no longer makes sense

import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10,5))

fspace = 1e3
fstart = 1e5
fstop = 200e6
w = srf.f2w(np.arange(fstart, fstop, fspace))

L = 22e-6
ESR = 3.7
SRF = 10e6
Qref = 15
fQref = 0.796e6

ZL = srf.L(L, w, ESR=ESR, Qref=Qref, fQref=fQref, srf=SRF)

plt.loglog(srf.w2f(w)/1.0e6, srf.mag(ZL))
plt.grid()
plt.gca().twinx()
plt.plot(srf.w2f(w)/1.0e6, srf.phase(ZL, deg = True), c = 'orange')
# plt.loglog(srf.w2f(w)/1.0e6, Q, c = 'purple')

plt.show()

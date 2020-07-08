import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

# f = np.logspace(5,9,1000)
# w = srf.f2w(f)

v1 = srf.Signal(10000, 0.1)
v1.make_tone(100, 0)

plt.subplot(2,1,1)
v1.plot_t(plt.gca())
plt.plot(v1.ts, -srf.Vf2Vt(v1.Vf, v1.ns))

plt.subplot(2,1,2)
plt.plot(v1.fs, srf.mag(v1.Vf))
# srf.plot_power_spectrum(plt.gca(), v1.ts, v1.Vt, time = True)

plt.show()

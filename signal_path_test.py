import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

# f = np.logspace(5,9,1000)
# w = srf.f2w(f)

v1 = srf.Signal(10000, 0.1)
v1.make_tone(100, 0)

stage1 = srf.Two_Port.from_network(v1.fs, [srf.L(1e-6, v1.fs)], [srf.C(1e-9, v1.fs)])

plt.subplot(2,1,1)
v1.plot_t(plt.gca())

plt.subplot(2,1,2)
plt.plot(v1.fs, srf.mag(v1.Vf))
# srf.plot_power_spectrum(plt.gca(), v1.ts, v1.Vt, time = True)

plt.show()

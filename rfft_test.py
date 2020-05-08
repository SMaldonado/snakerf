import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2


t = np.linspace(0,0.001,1000000)
f = 100000

v1 = srf.dBm2Vp(-100) * np.sin(srf.f2w(f) * t)
v2 = srf.Vt_noise(t)
v3 = srf.power_combine([v1,v2], t)

plt.subplot(4,1,1)
srf.plot_power_spectrum(plt.gca(), t, v1, time = True)
plt.subplot(4,1,2)
srf.plot_power_spectrum(plt.gca(), t, v2, time = True)
plt.subplot(4,1,3)
srf.plot_power_spectrum(plt.gca(), t, v3, time = True)
plt.subplot(4,1,4)
plt.plot(t, v3)
plt.show()

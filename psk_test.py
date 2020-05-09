import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2


t = np.linspace(0,0.039,100000)
f = 1000

fs = srf.fft_fs(t)
ws = srf.f2w(fs)

# print(srf.gold_codes(3))

#
v1 = srf.V_psk(t, f, f/10, [1,0,0,1], -100)
v2 = srf.Vt_noise(t)
v3 = srf.power_combine([v1,v2], t, out_Pf = True)
# print(srf.C(1e-9, ws))
R1 = 1e3
C1 = 10e-9
v4 = v3 * srf.Vdiv(srf.R(R1, ws), srf.C(C1, ws))

# print(srf.w2f(1/(R1*C1)))

# plt.subplot(4,1,1)
# srf.plot_power_spectrum(plt.gca(), t, v1, time = True)
# plt.subplot(4,1,2)
# srf.plot_power_spectrum(plt.gca(), t, v2, time = True)
plt.subplot(2,1,1)
srf.plot_power_spectrum(plt.gca(), fs, v3, time = False)
srf.plot_power_spectrum(plt.gca(), fs, v4, time = False)
plt.subplot(2,1,2)
plt.plot(t, srf.Pf2Vt(v3, len(t)))
plt.plot(t, srf.Pf2Vt(v4, len(t)))
plt.show()

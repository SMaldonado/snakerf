import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

data = '011110010001'
f = 2500000

f_bit = 10000
T_bit = 1/f_bit
h = 0.5
f_dev = h/(2*T_bit)
print(f_dev)

t = np.linspace(0,len(data)*T_bit - T_bit/100,100000)
fs = srf.fft_fs(t)
ws = srf.f2w(fs)

v1 = srf.V_fsk(t, f, f_bit, f_dev, data, -100)
v2 = srf.Vt_noise(t)
v3 = srf.power_combine([v1,v2], t, out_Pf = True)
# print(srf.C(1e-9, ws))
R1 = 1e3
C1 = 25e-12
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
for i in range(len(data)):
    plt.axvline(T_bit * i, ls = '--', c = 'black')
plt.show()

import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2


data = '011110010001'
n = 1
f = 25000

f_bit = 9001
T_bit = 1/f_bit

t = np.linspace(0,len(data)*T_bit/n - T_bit/100,100000)
fs = srf.fft_fs(t)
ws = srf.f2w(fs)

v1 = srf.V_psk(t, f, f_bit, data, -100, n = n)
v2 = srf.Vt_noise(t)
v3 = srf.power_combine([v1,v2], t, out_Pf = True)

R1 = 1e3
C1 = 50e-12
v4 = v3 * srf.Pdiv(srf.R(R1, ws), srf.C(C1, ws))

print(srf.w2f(1/(R1*C1)))

plt.subplot(2,1,1)
srf.plot_power_spectrum(plt.gca(), fs, v3, time = False)
srf.plot_power_spectrum(plt.gca(), fs, v4, time = False)
plt.subplot(2,1,2)
plt.plot(t, srf.Pf2Vt(v3, len(t)))
plt.plot(t, v1)
for i in range(len(data)):
    plt.axvline(T_bit * i, ls = '--', c = 'black')
plt.show()

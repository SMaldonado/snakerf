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

# v1, inverted, df, odd, even, plt_data = srf.V_msk(t, f, f_bit, data, -100)
v1 = srf.V_msk(t, f, f_bit, data, -100)
v2 = srf.Vt_noise(t)
v3 = srf.power_combine([v1,v2], t, out_Pf = True)

R1 = 1e3
C1 = 10e-9
v4 = v3 * srf.Vdiv(srf.R(R1, ws), srf.C(C1, ws))

plt.subplot(2,1,1)
srf.plot_power_spectrum(plt.gca(), t, v1, time = True)
# srf.plot_power_spectrum(plt.gca(), fs, v4, time = False)
plt.subplot(2,1,2)
plt.plot(t,v1)
# plt.plot(t,inverted, c = 'red')
# plt.plot(t,df*1.1, c = 'green')
# plt.plot(t,odd*1.2, c = 'purple')
# plt.plot(t,even*1.3, c = 'gold')
# plt.plot(t,plt_data*1.4, c = 'orange')

for i in range(len(data)):
    plt.axvline(T_bit * i, ls = '--', c = 'black')
plt.show()

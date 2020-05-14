import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

m = 5
# for code in srf.gold_codes(m):
#     print('{0:0{1:d}b}'.format(code,2**m - 1))


data = '{0:0{1:d}b}'.format(srf.gold_codes(m)[0], 2**m - 1)
print(data)
n = 1
f = 250000

f_bit = 9001
T_bit = 1/f_bit

t = np.linspace(0,len(data)*T_bit/n - T_bit/100,100000)
fs = srf.fft_fs(t)
ws = srf.f2w(fs)

v1 = srf.V_psk(t, f, f_bit, data, -90, n = n)
v2 = srf.V_msk(t, f, f_bit, data, -90)
# v3 = srf.V_psk(t, f, f_bit, data, -90, n = n)
vn = srf.Vt_noise(t)
v1n = srf.power_combine([v1,vn], t, out_Pf = True)
v2n = srf.power_combine([v2,vn], t, out_Pf = True)
# v3n = srf.power_combine([v3,vn], t, out_Pf = True)

R1 = 1e3
C1 = 25e-11
v1f = v1n * srf.Vdiv(srf.R(R1, ws), srf.C(C1, ws))
v2f = v2n * srf.Vdiv(srf.R(R1, ws), srf.C(C1, ws))
# v3f = v3n * srf.Vdiv(srf.R(R1, ws), srf.C(C1, ws))

# print(srf.w2f(1/(R1*C1)))

plt.subplot(2,1,1)
# srf.plot_power_spectrum(plt.gca(), fs, v3n, time = False)
srf.plot_power_spectrum(plt.gca(), t, v1, time = True)
srf.plot_power_spectrum(plt.gca(), t, v2, time = True)
plt.xlim(f-10*f_bit,f+10*f_bit)
plt.subplot(2,1,2)
# plt.plot(t, srf.Pf2Vt(v3f, len(t)))
plt.plot(t, srf.Pf2Vt(v1f, len(t)))
plt.plot(t, srf.Pf2Vt(v2f, len(t)))
for i in range(len(data)):
    plt.axvline(T_bit * i, ls = '--', c = 'black')
plt.show()

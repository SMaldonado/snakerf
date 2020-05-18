import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

m = 3
data = '{0:0{1:d}b}'.format(srf.gold_codes(m)[2], 2**m - 1)
print(data)
n = 1
f = 250000

f_bit = 9001
T_bit = 1/f_bit
t_max = len(data)*T_bit/n - T_bit/100
ns = 100000

V1 = srf.Signal(ns, t_max)
# V1.update_Vt(srf.V_psk(V1.ts, f, f_bit, data, -90, n = n))
V2 = V1.copy()

V1.add_noise(srf.NF2T_noise(3)*4)
V2.add_noise(srf.NF2T_noise(3))
V2.add_noise(srf.NF2T_noise(3))
V2.add_noise(srf.NF2T_noise(3))
V2.add_noise(srf.NF2T_noise(3))

print(srf.W2dBm(sum(srf.mag(V1.Pf))/max(V1.fs)))
print(srf.W2dBm(sum(srf.mag(V2.Pf))/max(V2.fs)))

srf.plot_power_spectrum(plt.gca(), V1.fs, V1.Pf)
srf.plot_power_spectrum(plt.gca(), V2.fs, V2.Pf)

plt.show()

# for i in range(len(data)):
#     plt.axvline(T_bit * i, ls = '--', c = 'black')
# plt.show()

# plt.subplot(2,1,1)
# # plt.plot(fs, srf.dB(srf.mag(srf.Pdiv(srf.R(R1, ws), srf.C(C1, ws)))))
# # plt.axvline(srf.w2f(1/(R1*C1)), ls = '-', c = 'black')
# # srf.plot_power_spectrum(plt.gca(), t, v1, time = True)
# # srf.plot_power_spectrum(plt.gca(), t, v2, time = True)
# # plt.xlim(f-10*f_bit,f+10*f_bit)
# plt.subplot(2,1,2)
# # plt.plot(t, srf.Pf2Vt(v3f, len(t)))
# plt.plot(t, srf.Pf2Vt(v1f, len(t)))
# plt.plot(t, srf.Pf2Vt(v2f, len(t)))
# for i in range(len(data)):
#     plt.axvline(T_bit * i, ls = '--', c = 'black')
# plt.show()

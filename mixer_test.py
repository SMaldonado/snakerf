import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

from scipy import signal
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html#scipy.signal.periodogram

m = 5
data = '{0:0{1:d}b}'.format(srf.gold_codes(m)[2], 2**m - 1)
print(data)
n = 1
f = 5e6

f_bit = 9001
T_bit = 1/f_bit
t_max = len(data)*T_bit/n - T_bit/100
fs = 10e6
ns = 100000
# t_max = ns/fs


V1 = srf.Signal(ns, t_max)
# V1.make_tone(10000, 0)
V1.update_Vt(srf.V_msk(V1.ts, 25000, f_bit, data, 0))

V2 = srf.Signal(ns, t_max)
V2.make_tone(100000, 0.1)

mx = srf.Mixer()
V3 = mx.mix(V1, V2)

plt.subplot(2,1,1)
V1.plot_f(plt.gca())
V2.plot_f(plt.gca())
V3.plot_f(plt.gca())

plt.subplot(2,1,2)
plt.plot(V1.ts, V1.Vt)
plt.plot(V2.ts, V2.Vt)
plt.plot(V3.ts, V3.Vt)

plt.show()
#
# gain_max = 40
# f_c = f
# f_clo = f_c * 0.95
# f_chi = f_c * 1.05
# slope_dec = -40
#
# dB_gain = [gain_max + slope_dec/10, gain_max, gain_max, gain_max + slope_dec/10]
# f_gain = [f_clo * (10**-0.1), f_clo, f_chi, f_chi * (10**0.1)]
#
# amp = srf.Amplifier(1, dB_gain, f_gain)
#
# plt.subplot(2,2,1)
# srf.plot_power_spectrum(plt.gca(), V1.fs, V1.Pf)
# H = amp.gain(V1.fs)
# plt.semilogx(V1.fs, H)
# plt.axhline( gain_max + slope_dec, c = 'k', ls = '--')
# plt.axhline( gain_max + 2*slope_dec, c = 'k', ls = '--')
# plt.axhline( gain_max + 3*slope_dec, c = 'k', ls = '--')
# plt.axvline(f_clo/10, c = 'k', ls = '--')
# plt.axvline(f_clo/100, c = 'k', ls = '--')
# plt.axvline(f_clo/1000, c = 'k', ls = '--')
# plt.axvline(f_chi*10, c = 'k', ls = '--')
# V1.amplify(amp)
# srf.plot_power_spectrum(plt.gca(), V1.fs, V1.Pf)
#
# plt.subplot(2,2,3)
# plt.plot(V1.ts, V1.Vt)
# # plt.show()
#
#
# V2 = srf.Signal(ns, t_max)
# # V2.update_Vt(srf.dBm2Vp(-100) * np.sin(2*pi*f*V2.ts) + srf.Vt_background_noise(V2.ts, V2.fs))
# V2.update_Vt(srf.V_psk(V2.ts, f, f_bit, data, -70) + srf.Vt_background_noise(V2.ts, V2.fs))
#
# plt.subplot(2,2,2)
# srf.plot_power_spectrum(plt.gca(), V2.fs, V2.Pf)
# plt.subplot(2,2,4)
# # plt.plot(V2.ts, V2.Vt)
#
# V2.amplify(amp)
#
# plt.subplot(2,2,2)
# srf.plot_power_spectrum(plt.gca(), V2.fs, V2.Pf)
# plt.subplot(2,2,4)
# plt.plot(V2.ts, V2.Vt)
# plt.xlim(0, 10/f)
#
#
#
# plt.show()

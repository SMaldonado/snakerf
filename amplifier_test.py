import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

from scipy import signal
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html#scipy.signal.periodogram

m = 3
data = '{0:0{1:d}b}'.format(srf.gold_codes(m)[2], 2**m - 1)
print(data)
n = 1
f = 22e3

f_bit = 9001
T_bit = 1/f_bit
# t_max = len(data)*T_bit/n - T_bit/100
fs = 1e5
ns = 100000
t_max = ns/fs


V1 = srf.Signal(ns, t_max)
V1.update_Pf(srf.Vt_background_noise(V1.ts, V1.fs))

amp = srf.Amplifier(0.01, dB_gain = [-10, 20, 20, 0], f_gain = [18e3, 20e3, 25e3, 27e3])

plt.subplot(2,1,1)
srf.plot_power_spectrum(plt.gca(), V1.fs, V1.Pf)
plt.plot(V1.fs, amp.gain(V1.fs))
V1.amplify(amp)
srf.plot_power_spectrum(plt.gca(), V1.fs, V1.Pf)

plt.subplot(2,1,2)
plt.plot(V1.ts, V1.Vt)
plt.show()

# V2 = srf.Signal(ns, t_max)
# V2.add_noise()
#
# print(srf.NF2T_noise(3))
# srf.plot_power_spectrum(plt.gca(), V2.fs, V2.Pf)
#
# f_ref =  [0, 4, 5, 8.3, 12] # log frequency
# Fa_ref = [270, 150, 80, 0, 0] # Fa = 10*log10(T_noise/t0)
#
# V1.update_Pf(srf.Vt_background_noise(V1.ts, V1.fs))
# srf.plot_power_spectrum(plt.gca(), V1.fs, V1.Pf)
#
# T_noise = srf.undB(np.interp(np.log10(np.maximum(V1.fs,np.ones(len(V1.fs)))), f_ref, Fa_ref)) * srf.t0 # weird thing with ones to avoid log(0)
# plt.plot(V1.fs, srf.W2dBm(4*srf.kB*T_noise*V1.df))
#
# N = 100
# moving_avg = np.convolve(srf.mag(V1.Pf * V1.Z0 / V1.df), np.ones((N,))/N, mode='valid') * V1.df/V1.Z0
# plt.plot(V1.fs[:-N+1], srf.W2dBm(moving_avg))
#
# moving_avg = np.convolve(srf.mag(V2.Pf * V2.Z0 / V2.df), np.ones((N,))/N, mode='valid') * V2.df/V2.Z0
# plt.plot(V2.fs[:-N+1], srf.W2dBm(moving_avg))


plt.show()

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
f = 1234

f_bit = 9001
T_bit = 1/f_bit
# t_max = len(data)*T_bit/n - T_bit/100
fs = 10e3
ns = 100000
t_max = ns/fs


V1 = srf.Signal(ns, t_max)
# # V1.update_Vt(amp*np.sin(2*np.pi*freq*time))
# V1.add_noise(noise = 0.001 /(4 * srf.kB * V1.Z0))
# srf.plot_power_spectrum(plt.gca(), V1.fs, V1.Pf)
# mean_noise = np.mean(srf.mag(V1.Pf * V1.Z0 / V1.df))
# print(mean_noise)
# plt.axhline(srf.W2dBm(mean_noise * V1.df/V1.Z0), c = 'k', ls = '--')


# srf.plot_power_spectrum(plt.gca(), V1.ts, np.random.normal(0, 1, len(V1.ts)), time = True)

f_ref =  [0, 4, 5, 8.3, 12] # log frequency
Fa_ref = [270, 150, 80, 0, 0] # Fa = 10*log10(T_noise/t0)

V1.update_Pf(srf.Vt_background_noise(V1.ts, V1.fs))
srf.plot_power_spectrum(plt.gca(), V1.fs, V1.Pf)

T_noise = srf.undB(np.interp(np.log10(np.maximum(V1.fs,np.ones(len(V1.fs)))), f_ref, Fa_ref)) * srf.t0 # weird thing with ones to avoid log(0)
plt.plot(V1.fs, srf.W2dBm(srf.kB*T_noise*V1.df))

N = 100
moving_avg = np.convolve(srf.mag(V1.Pf * V1.Z0 / V1.df), np.ones((N,))/N, mode='valid') * V1.df/V1.Z0
plt.plot(V1.fs[:-N+1], srf.W2dBm(moving_avg))


plt.show()

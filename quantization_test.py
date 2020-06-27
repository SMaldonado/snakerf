import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

# f = np.logspace(5,9,1000)
# w = srf.f2w(f)

fc = 10000
f_sym = 1000
f_dev = 500
f_sample = 25000
random_data = '100110111010101101100110111010101101100110111010101101100110111010101101100110111010101101'

v1 = srf.Signal(10000, 0.05)
v1.update_Vt(srf.V_fsk(v1.ts, fc, f_sym, f_dev, random_data, -120, n = 1))
v1.add_noise()

y1 = srf.demod_fsk(v1.Vt, v1.ts, fc, f_sym, f_dev, f_sample = f_sample)#, quantize_func = srf.quantize_adc, V_full = 1, n_bits = 10)
y2 = srf.demod_fsk(v1.Vt, v1.ts, fc, f_sym, -f_dev, f_sample = f_sample)#, quantize_func = srf.quantize_adc, V_full = 1, n_bits = 10)

plt.subplot(2,1,1)
plt.plot(np.arange(min(v1.ts), max(v1.ts), 1/f_sym) + 0.5/f_sym, srf.mag(y1))
plt.plot(np.arange(min(v1.ts), max(v1.ts), 1/f_sym) + 0.5/f_sym, srf.mag(y2))

for i in range(len(random_data)):
    plt.axvline((i+1)*(1/f_sym), color = 'black', ls = '--')

plt.xlim(min(v1.ts), max(v1.ts))

plt.subplot(2,1,2)

plt.plot(v1.ts, v1.Vt)
# srf.plot_power_spectrum(plt.gca(), v1.ts, v1.Vt, time = True)

# for i in range(len(random_data)):
#     plt.axvline((i+1)*(1/f_sym), color = 'black', ls = '--')
#
# plt.xlim(min(v1.ts), max(v1.ts))

# plt.plot(v1.ts, srf.quantize_ideal(v1.Vt))
# plt.plot(v1.ts, srf.quantize_adc(v1.Vt + srf.dBm2Vp(0), 2*srf.dBm2Vp(0), 2))
# v1.plot_t(plt.gca())
# plt.plot(v1.ts, -srf.Vf2Vt(v1.Vf, v1.ns))

plt.show()

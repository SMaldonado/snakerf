import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

# f = np.logspace(5,9,1000)
# w = srf.f2w(f)

v1 = srf.Signal(10000, 0.05)
v1.update_Vt(srf.V_fsk(v1.ts, 10000, 1000, 2000, '100110111010101101100110111010101101100110111010101101100110111010101101100110111010101101', 0, n = 1))

print(srf.demod_fsk(v1.Vt, v1.ts, 10000, 1000, 2000, quantize_func = srf.quantize_adc, V_full = 1, n_bits = 10))

plt.plot(v1.ts, srf.quantize_ideal(v1.Vt))
plt.plot(v1.ts, srf.quantize_adc(v1.Vt + srf.dBm2Vp(0), 2*srf.dBm2Vp(0), 2))
# v1.plot_t(plt.gca())
# plt.plot(v1.ts, -srf.Vf2Vt(v1.Vf, v1.ns))

plt.show()

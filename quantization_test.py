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
m = 9
random_data = '{0:0{1:d}b}'.format(srf.gold_codes(m)[2], 2**m - 1) + '0'
P_dBm = -120

v1 = srf.Signal(10000, 0.05)
v1.update_Vt(srf.V_fsk(v1.ts, fc, f_sym, f_dev, random_data, P_dBm, n = 2))
v1.add_noise()

V_pk = srf.dBm2Vp(P_dBm, 50)

syms, p_syms = srf.demod_fsk(v1.Vt, v1.ts, fc, f_sym, f_dev, n = 2, f_sample = f_sample, quantize_func = srf.quantize_adc, V_full = V_pk, n_bits = 12)

# print(random_data[:len(syms)])
# print(syms)
errs = ''.join(['0' if syms[i] == random_data[i] else '1' for i in range(len(syms))])
# print(errs)
print('{} / {}'.format(errs.count('1'), len(syms)))

plt.subplot(2,1,1)
plt.plot(np.arange(min(v1.ts), max(v1.ts), 1/f_sym) + 0.5/f_sym, p_syms)

for i in range(len(random_data)):
    plt.axvline((i+1)*(1/f_sym), color = 'black', ls = '--')

plt.xlim(min(v1.ts), max(v1.ts))

plt.subplot(2,1,2)
plt.plot(v1.ts, v1.Vt)
plt.show()

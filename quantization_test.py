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
n = 2

test_bits = 500
f_sim = 2e5
t_sim = test_bits / (f_sym * n)

BW = f_sim/2
P_sig = srf.dBm2W(P_dBm)/(2**n)
P_noise = srf.kB * srf.t0 * BW
print(srf.W2dBm(P_noise))

BW_c = (f_dev * (2**n)) + (2 * f_sym)
bitrate = f_sym

Eb_N0 = P_sig * BW_c / (P_noise * bitrate)
print(srf.dB(Eb_N0))

V_pk = srf.dBm2Vp(P_dBm, 50)

n_errs = 0
n_tests = 10
for i in range(n_tests):

    v1 = srf.Signal(f_sim * t_sim, t_sim)
    v1.update_Vt(srf.V_fsk(v1.ts, fc, f_sym, f_dev, random_data, P_dBm, n = n))
    v1.add_noise()

    syms, p_syms = srf.demod_fsk(v1.Vt, v1.ts, fc, f_sym, f_dev, n = n, f_sample = f_sample)#, quantize_func = srf.quantize_adc, V_full = V_pk, n_bits = 12)

    # print(random_data[:len(syms)])
    # print(syms)
    errs = ''.join(['0' if syms[i] == random_data[i] else '1' for i in range(len(syms))])
    print('{} / {}'.format(errs.count('1'), len(syms)))
    n_errs = n_errs + errs.count('1')
    # print(errs)

print('{} / {} ({:e})'.format(n_errs, n_tests * test_bits, n_errs/(n_tests * test_bits)))


###############

srf.plot_power_spectrum(plt.gca(), v1.ts, v1.Vt, time = True)
plt.show()

###############

# plt.subplot(2,1,1)
# plt.plot(np.arange(min(v1.ts), max(v1.ts), 1/f_sym) + 0.5/f_sym, p_syms)
#
# for i in range(len(random_data)):
#     plt.axvline((i+1)*(1/f_sym), color = 'black', ls = '--')
#
# plt.xlim(min(v1.ts), max(v1.ts))
#
# plt.subplot(2,1,2)
# plt.plot(v1.ts, v1.Vt)
# plt.show()

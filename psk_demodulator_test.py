import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2

# f = np.logspace(5,9,1000)
# w = srf.f2w(f)

fc = 10000
f_sym = 1000
f_dev = 0
f_sample = 25000
m = 9
random_data = '{0:0{1:d}b}'.format(srf.gold_codes(m)[2], 2**m - 1) + '0'
P_dBm = -123
n = 2

test_bits = 500
f_sim = 2e5
t_sim = test_bits / (f_sym * n)

BW = f_sim
P_sig = srf.dBm2W(P_dBm)
print(P_sig)
P_noise = srf.kB * srf.t0 * BW
print(srf.W2dBm(P_noise))

BW_c = f_sample / 2
bitrate = f_sym * n

Eb_N0 = P_sig * BW_c / (P_noise * bitrate)
print(srf.dB(Eb_N0))

V_pk = srf.dBm2Vp(P_dBm, 50)

n_errs = 0
n_tests = 25
for i in range(n_tests):

    v1 = srf.Signal(f_sim * t_sim, t_sim)
    v1.update_Vt(srf.V_psk(v1.ts, fc, f_sym, random_data, P_dBm, n = n))
    v1.add_noise()

    data, syms, p_syms = srf.demod_psk(v1.Vt, v1.ts, fc, f_sym, n = n, f_sample = f_sample)#, quantize_func = srf.quantize_adc, V_full = V_pk, n_bits = 12)

    errs = ''.join(['0' if data[i] == random_data[i] else '1' for i in range(len(data))])
    print('{} / {}'.format(errs.count('1'), len(data)))
    n_errs = n_errs + errs.count('1')

print('{} / {} ({:e})'.format(n_errs, n_tests * test_bits, n_errs/(n_tests * test_bits)))

###############

# srf.plot_power_spectrum(plt.gca(), v1.ts, v1.Vt, time = True)
# plt.show()

###############

# plt.subplot(2,1,1)
# plt.plot(np.arange(min(v1.ts), max(v1.ts), 1/f_sym) + 0.5/f_sym, syms)
# # plt.plot(np.arange(min(v1.ts), max(v1.ts), 1/f_sample) + 0.5/f_sample, i)
# # plt.plot(np.arange(min(v1.ts), max(v1.ts), 1/f_sample) + 0.5/f_sample, q)
# # plt.plot(np.arange(min(v1.ts), max(v1.ts), 1/f_sym) + 0.5/f_sym, srf.phase(p_syms))
#
# for i in range(len(random_data)):
#     plt.axvline((i+1)*(1/f_sym), color = 'black', ls = '--')
#
# q = 2**n
# d_phi = 360/q # get phase step
# T_sym = 1/f_sym # get symbol time
#
# for i in range(q):
#     x = 1 if i>=q/2 else 0
#     plt.axhline(i - q/2 + x, color = 'black', ls = '--')
#
#
#
# plt.xlim(min(v1.ts), max(v1.ts))
# plt.title(' '.join([random_data[i:i+n] for i in range(0, test_bits, n)]))
#
# plt.subplot(2,1,2)
# # plt.plot(v1.ts, v1.Vt)
# srf.plot_power_spectrum(plt.gca(), v1.ts, v1.Vt, time = True)
# plt.show()

import snakerf as srf
import matplotlib.pyplot as plt
import numpy as np
from math import inf, pi, log2, ceil

# f = np.logspace(5,9,1000)
# w = srf.f2w(f)

fc = 10002
f_sym = 1000
f_dev = 0
m = 9
random_data = '{0:0{1:d}b}'.format(srf.gold_codes(m)[5], 2**m - 1) + '0'
P_dBm = 0
n = 4

test_bits = 500
f_sim = 2e5
t_sim = test_bits / (f_sym * n)
v1 = srf.Signal(f_sim * t_sim, t_sim)

v_qam = srf.V_qam(v1.ts, fc, f_sym, random_data, P_dBm, n = n)
t_sym_sample = np.arange(0, t_sim, 1/f_sym)
v_qam_sample = np.array(np.interp(t_sym_sample, v1.ts, v_qam))

f, (ax1, ax2, ax3) = plt.subplots(3,1)#,sharex = True)
ax1.plot(v1.ts, v_qam)

samples_sym = f_sim * t_sim / (test_bits/n)

v_qam_iq = v_qam * (np.cos(srf.f2w(fc)*v1.ts) + 1j*np.sin(srf.f2w(fc)*v1.ts))
v_qam_real = [np.mean([v_qam_iq[x].real for x in range(int(ceil(samples_sym*i)), int(ceil(samples_sym*(i+1))), 1)]) for i in range(test_bits//n)] # get real mag per signal period
v_qam_imag = [np.mean([-1 * v_qam_iq[x].imag for x in range(int(ceil(samples_sym*i)), int(ceil(samples_sym*(i+1))), 1)]) for i in range(test_bits//n)] # get imag mag per signal period
# negative because j**2 = -1

symsi_demod = [round(x/(srf.dBm2Vrms(P_dBm - 6) / (2 ** ((n/2) - 1)))) for x in v_qam_real]
symsq_demod = [round(x/(srf.dBm2Vrms(P_dBm - 6) / (2 ** ((n/2) - 1)))) for x in v_qam_imag]

bitsi_demod = srf.sym2data(symsi_demod, n//2)
bitsq_demod = srf.sym2data(symsq_demod, n//2)

data_demod = ''.join([i + q for i,q in zip(bitsi_demod.split(' '), bitsq_demod.split(' '))])

print(random_data[0:100])
print(data_demod[0:100])

symsi_demod = [round(x/0.022) for x in v_qam_real]
ax2.plot(v_qam_iq.real, c = 'orange')
ax2.plot(-v_qam_iq.imag, c = 'green')

ax3.plot(v_qam_real , c = 'orange')
ax3.plot(v_qam_imag , c = 'green')

plt.show()

# plt.scatter(srf.mag(v_qam) * np.cos(np.angle(v_qam)), srf.mag(v_qam) * np.sin(np.angle(v_qam)))
# plt.gca().set_aspect('equal')
# plt.scatter(np.cos(srf.f2w(fc*t_sym_sample)) * v_qam_sample, np.sin(srf.f2w(fc*t_sym_sample)) * v_qam_sample)




#
# test_bits = 500
# f_sim = 2e5
# t_sim = test_bits / (f_sym * n)
#
# BW = f_sim / 2
# P_sig = srf.dBm2W(P_dBm)
# print(P_sig)
# P_noise = srf.kB * srf.t0 * BW
# print(srf.W2dBm(P_noise))
#
# BW_c = f_sample / 2
# bitrate = f_sym * n
#
# Eb_N0 = P_sig * BW_c / (P_noise * bitrate)
# print(srf.dB(Eb_N0))
#
# V_pk = srf.dBm2Vp(P_dBm, 50)
#
# n_errs = 0
# n_tests = 25
# for i in range(n_tests):
#
#     v1 = srf.Signal(f_sim * t_sim, t_sim)
#     v1.update_Vt(srf.V_psk(v1.ts, fc, f_sym, random_data, P_dBm, n = n))
#     v1.add_noise()
#
#     data, syms, p_syms = srf.demod_psk(v1.Vt, v1.ts, fc, f_sym, n = n, f_sample = f_sample)#, quantize_func = srf.quantize_adc, V_full = V_pk, n_bits = 12)
#
#     errs = ''.join(['0' if data[i] == random_data[i] else '1' for i in range(len(data))])
#     print('{} / {}'.format(errs.count('1'), len(data)))
#     n_errs = n_errs + errs.count('1')
#
# print('{} / {} ({:e})'.format(n_errs, n_tests * test_bits, n_errs/(n_tests * test_bits)))

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

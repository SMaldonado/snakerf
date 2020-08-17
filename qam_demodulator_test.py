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
P_dBm = -120
n = 4

test_bits = 500
f_sim = 2e5
t_sim = test_bits / (f_sym * n)
v1 = srf.Signal(f_sim * t_sim, t_sim)

v_qam = srf.V_qam(v1.ts, fc, f_sym, random_data, P_dBm, n = n) + srf.Vt_thermal_noise(v1.ts, v1.fs)
t_sym_sample = np.arange(0, t_sim, 1/f_sym)
v_qam_sample = np.array(np.interp(t_sym_sample, v1.ts, v_qam))

# f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)#,sharex = True)
f = plt.figure(figsize = (12,6))
gs = f.add_gridspec(3, 6)
ax1 = f.add_subplot(gs[0,:3])
ax2 = f.add_subplot(gs[1,:3])
ax3 = f.add_subplot(gs[2,:3])
ax4 = f.add_subplot(gs[:,3:])

ax1.plot(v1.ts, v_qam)

samples_sym = f_sim * t_sim / (test_bits/n)

v_qam_iq = v_qam * (np.cos(srf.f2w(fc)*v1.ts) + 1j*np.sin(srf.f2w(fc)*v1.ts))
v_qam_real = np.array([np.mean([v_qam_iq[x].real for x in range(int(ceil(samples_sym*i)), int(ceil(samples_sym*(i+1))), 1)]) for i in range(test_bits//n)]) # get real mag per signal period
v_qam_imag = np.array([np.mean([-1 * v_qam_iq[x].imag for x in range(int(ceil(samples_sym*i)), int(ceil(samples_sym*(i+1))), 1)]) for i in range(test_bits//n)]) # get imag mag per signal period
# negative because j**2 = -1

symsi_demod = [round(x/(srf.dBm2Vrms(P_dBm - 6) / (2 ** ((n/2) - 1)))) for x in v_qam_real]
symsq_demod = [round(x/(srf.dBm2Vrms(P_dBm - 6) / (2 ** ((n/2) - 1)))) for x in v_qam_imag]

# bitsi_demod = srf.sym2data(symsi_demod, n//2)
# bitsq_demod = srf.sym2data(symsq_demod, n//2)
#
# data_demod = ''.join([i + q for i,q in zip(bitsi_demod.split(' '), bitsq_demod.split(' '))])
#
# print(random_data[0:100])
# print(data_demod[0:100])
#
# ax2.plot(v_qam_iq.real, c = 'orange')
# ax2.plot(-v_qam_iq.imag, c = 'green')
#
# ax3.plot(v_qam_real , c = 'orange')
# ax3.plot(v_qam_imag , c = 'green')

# ax4.scatter(symsi_demod - 0.5*np.sign(symsi_demod), symsq_demod - 0.5*np.sign(symsq_demod))
mag_v_qam = np.sqrt(v_qam_real**2 + v_qam_imag**2) / (srf.dBm2Vrms(P_dBm - 6) / (2 ** ((n/2) - 1)))
phase_v_qam = srf.phase(v_qam_real + 1j*v_qam_imag)
print(max(mag_v_qam))
ax4.scatter(mag_v_qam * np.cos(phase_v_qam), mag_v_qam * np.sin(phase_v_qam))
ax4.set_aspect('equal')

plt.show()


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

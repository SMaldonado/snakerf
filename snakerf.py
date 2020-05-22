import math
from math import pi, inf
import numpy as np
from numpy import log10, log2, rad2deg, deg2rad, sqrt
import matplotlib.ticker as ticker
from copy import deepcopy

c = 299792458.0 # speed of light in vacuum, m/s
kB = 1.380649E-23 # Boltzmann constant, J/K
t0 = 290 # noise figure temperature, Kelvin

@np.vectorize
def par(Z1, Z2):
    if Z1 == inf: return Z2
    if Z2 == inf: return Z1
    if Z1+Z2 == 0: return inf
    return (Z1 * Z2)/(Z1 + Z2)

def ser(Z1, Z2):
    return Z1 + Z2

def mag(x):
    return np.abs(x)

def phase(x, unwrap = True, deg = False):
    if unwrap: angle = np.unwrap(np.angle(x))
    else: angle = np.angle(x)

    if deg: return rad2deg(angle)
    else: return angle

# passive component frequency responses

# @np.vectorize
def C_proto(C, w, ESR = 0):
    if C == inf: return ESR
    if C == 0 or w == 0: return inf
    return (1/(1j*w*C)) + ESR

C = np.vectorize(C_proto, otypes = [np.complex])


def L(L, w, ESR = 0, Cpar = 0, Gpar = 0, Qref = inf, fQref = inf, srf = inf, Zsrf = inf):

    # Inductor model assumes the following parasitics:
    #     |----Gpar----|
    #     |----Cpar----|
    #  ---|---L--ESR---|---
    #
    # Cpar can be provided or calculated from the self-resonant frequency
    # Gpar can be provided or calculated from a known Q at a known frequency
    # (or a known maximum impedance at resonance)

    Cpar_eff = Cpar
    Gpar_eff = Gpar

    if Gpar_eff == 0 and Qref != inf and fQref != inf: # calculate Gpar from measured Q
        wQref = f2w(fQref)

        if ESR == 0:
            Lp = L
            Rp = Qref * wQref * Lp
            Gpar_eff = 1/Rp
        else: # transform series RL to parallel RL
            Qs = wQref*L/ESR
            Lp = L * (1 + Qs**2.0) / (Qs**2.0)
            ESRp = (1 + Qs**2.0)*ESR
            Rp = Qref * wQref * Lp
            Gpar_eff = (1/Rp) - (1/ESRp)

    if Cpar_eff == 0 and srf != inf:
        wsrf = f2w(srf)

        if ESR == 0:
            Lp = L
            if Zsrf != inf and Gpar_eff == 0:
                Gpar_eff = 1/Zsrf

        else: # transform series RL to parallel RL
            Qs = wsrf*L/ESR
            Lp = L * (1 + Qs**2.0) / (Qs**2.0)
            ESRp = (1 + Qs**2.0)*ESR
            if Zsrf != inf and Gpar_eff == 0:
                Gpar_eff = (1/Zsrf) - (1/ESRp)

        Cpar_eff = 1/(Lp * wsrf**2.0)

    return par(1j*w*L + ESR, par(C(Cpar_eff, w), G(Gpar_eff, w)))

def R(R, w):
    return R * np.ones(len(w))

def G(G, w):
    if G == 0: return inf * np.ones(len(w))
    return (1.0/G) * np.ones(len(w))

def Zopen(w):
    return inf * np.ones(len(w))

# dB handling

def dB(x): # linear power gain to dB power gain
    return 10*log10(x)

def undB(x): # dB power gain to linear power gain
    return 10.0**(x/10.0)

def dBv(x): # linear voltage gain to dB gain
    return 20*log10(mag(x))

def undBv(x): # dB gain to linear voltage gain
    return 10.0**(x/20.0)

# Unit conversions

def f2w(x): # Hz to Rad/s
    return 2.0*pi*x

def w2f(x): # Rad/s to Hz
    return x/(2.0*pi)

def W2dBm(W): # power [W] to power [dBm]
    return dB(W) + 30.0

def dBm2W(dBm): # power [dBm] to power [W]
    return undB(dBm - 30.0)

def Vp2dBm(Vp,  Z0 = 50): # sine wave voltage amplitude [V] to power [dBm]
    return W2dBm(Vp**2.0 / (2.0*Z0))

def dBm2Vp(dBm,  Z0 = 50): # power [dBm] to sine wave voltage amplitude [V]
    return sqrt(dBm2W(dBm) * 2.0 * Z0)

def dBm2Vrms(dBm,  Z0 = 50): # power [dBm] to sine wave rms voltage [V]
    return dBm2Vp(dBm, Z0)/sqrt(2)

def Vrms2dBm(Vrms,  Z0 = 50): # sine wave rms voltage [V] to power [dBm]
    return Vp2dBm(Vrms * sqrt(2), Z0)

# Spectra handling

def fft_fs(t_sample):
    timestep = (max(t_sample) - min(t_sample))/len(t_sample)
    return np.fft.rfftfreq(len(t_sample), d = timestep)

def Vf2Pf(Vf, Z0 = 50): # f-domain voltage to f-domain power
    return mag(Vf) * (Vf.real + 1j*Vf.imag) / (2*Z0)

def Pf2Vf(Pf, Z0 = 50): # f-domain power to f-domain voltage
    magnitude = sqrt(mag(Pf) * 2*Z0)
    phase = Pf / mag(Pf)

    return magnitude * (phase.real + 1j*phase.imag)

def Vt2Vf(Vt, ns): # time-domain voltage to f-domain voltage
    # see https://www.sjsu.edu/people/burford.furman/docs/me120/FFT_tutorial_NI.pdf
    # see also: http://www.cmp.caltech.edu/~mcc/Chaos_Course/Lesson6/Power.pdf#page=7&zoom=100,0,0
    return np.fft.rfft(Vt)* 2/ns # numpy fft scaling; see https://numpy.org/doc/stable/reference/routines.fft.html#normalization

def Vf2Vt(Vf, ns): # f-domain voltage to time-domain voltage
    return np.fft.irfft(Vf * ns/2).real # numpy fft scaling; see https://numpy.org/doc/stable/reference/routines.fft.html#normalization

def Vt2Pf(Vt, ns, Z0 = 50): # time-domain voltage to f-domain power
    return Vf2Pf(Vt2Vf(Vt, ns), Z0)

def Pf2Vt(Pf, ns, Z0 = 50): # f-domain power to time-domain voltage
    return Vf2Vt(Pf2Vf(Pf, Z0), ns)

def power_combine(Vts, ts, Z0 = 50, out_Pf = False): # power combine array of time-domain voltages Vts
    ns = len(ts)
    Pf = np.sum([Vt2Pf(Vt, ns) for Vt in Vts], axis = 0) # sum elementwise

    if out_Pf: return Pf
    return Pf2Vt(Pf, ns, Z0)

# Voltage noise

def NF2T_noise(NF, dB = True): # convert noise figure in dB (or linear noise factor for dB = False) to noise temperature (in K)
    # see http://literature.cdn.keysight.com/litweb/pdf/5952-8255E.pdf eqn 1-5
    if dB: F = undB(NF)
    else: F = NF

    return t0*(F-1)

def Vt_thermal_noise(ts, fs, T_noise = t0, Z0 = 50, out_Pf = False): # create sampled time-domain additive white Gaussian voltage noise of specified noise temperature
    # see: https://www.ti.com/lit/an/slva043b/slva043b.pdf
    # see: https://en.wikipedia.org/wiki/Noise_temperature
    # see: https://www.ietlabs.com/pdf/GR_Appnote/IN-103%20Useful%20Formulas,%20Tables%20&%20Curves%20for.pdf
    # see: https://en.wikipedia.org/wiki/Noise_spectral_density
    # see: https://training.ti.com/system/files/docs/1312%20-%20Noise%202%20-%20slides.pdf
    # see: https://electronics.stackexchange.com/questions/303337/fourier-transform-of-additive-white-gaussian-noise
    # see: https://www.gaussianwaves.com/2013/11/simulation-and-analysis-of-white-noise-in-matlab/   --- explains non-flat spectrum
    # tl;dr: appropriately mathematically representing white noise is hard. Buyer beware.

    # Noise voltage variance (usually ̅V^2) as a single-sided spectral density usually equals:
    # ̅V^2/B = 4*kB*T*R
    V2_noise_Hz = 4 * kB*T_noise*mag(Z0)

    # fill full sampling bandwidth of t_sample with white noise - note that this may not always be desired
    f_nyq = max(fs)
    V_stddev_noise = sqrt(V2_noise_Hz * f_nyq)
    noise = np.random.normal(0, V_stddev_noise, len(ts))

    if out_Pf: return Vt2Pf(noise, len(ts), Z0 = Z0)
    else: return noise

def Vt_background_noise(ts, fs, Z0 = 50):
    # Atmospheric background noise:
    # http://www.dtic.mil/dtic/tr/fulltext/u2/a359931.pdf
    # https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.372-7-200102-S!!PDF-E.pdf

    # Non-white noise generally:
    # https://www.tandfonline.com/doi/pdf/10.1080/13873954.2017.1298622
    # https://groups.google.com/forum/#!topic/comp.dsp/bEwaXTMTmjM
    # https://www.researchgate.net/post/How_do_I_generate_time_series_data_from_given_PSD_of_random_vibration_input
    # https://dsp.stackexchange.com/questions/22587/generate-time-domain-random-signal-from-psd

    # low-resolution model of background noise temperature
    f_ref =  [0, 4, 5, 8.3, 12] # log frequency
    Fa_ref = [270, 150, 80, 0, 0] # Fa = 10*log10(T_noise/t0)

    T_noise = undB(np.interp(log10(np.maximum(fs,np.ones(len(fs)))), f_ref, Fa_ref)) * t0 # weird thing with ones to avoid log(0)
    V2_noise_Hz = 4*kB*T_noise*mag(Z0)
    V2_var_noise = np.trapz(V2_noise_Hz, fs) # integrate total noise power
    V_stddev_noise = sqrt(V2_var_noise)
    V_noise_white = np.random.normal(0, 1, len(ts))

    Pf_noise_white = Vt2Pf(V_noise_white, len(ts), Z0 = Z0)
    H_norm = V2_noise_Hz * (fs[1]-fs[0])
    Pf_noise = Pf_noise_white * H_norm

    # print(V2_var_noise)
    # print(np.mean(H_norm))
    # print(np.var(Pf2Vt(Pf_noise, len(ts), Z0)))

    return Pf_noise

# signal class
# TODO: lots

class Signal: # represents a nodal voltage in a given characteristic impedance
    def __init__(self, ns, t_max, sig = 0, sig_Vt = True, Z0 = 50):
        self.ns = ns
        self.t_max = t_max
        self.Z0 = Z0

        self.dt = t_max/ns
        self.ts = make_time(ns, t_max)
        self.fs = fft_fs(self.ts)

        if sig == 0: # initialize to zero signal
            self.Vt = np.zeros(len(self.ts))
            self.Pf = np.zeros(len(self.fs))
        else:
            if sig_Vt: # initialize to provided Vt
                self.update_Vt(sig)
            else: # initialize to provided Pf
                self.update_Pf(sig)

    def update_Vt(self, Vt): # update time-domain voltage and ensure f-domain consistency
        if len(Vt) != len(self.ts): pass # TODO: throw exception
        self.Vt = Vt
        self.Pf = Vt2Pf(self.Vt, self.ns, self.Z0)

    def update_Pf(self, Pf): # update power spectrum and ensure time-domain consistency
        if len(Pf) != len(self.fs): pass # TODO: throw exception
        self.Pf = Pf
        self.Vt = Pf2Vt(self.Pf, self.ns, self.Z0)

    def add_noise(self, noise = t0, NF = False):
        if NF: T_noise = NF2T_noise(noise)
        else: T_noise = noise

        # EXPERIMENTAL: Add noise by directly summing voltages, not power combining
        # see https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
        self.update_Vt(self.Vt + Vt_thermal_noise(self.ts, self.fs, T_noise = T_noise, Z0 = self.Z0))

    def copy(self):
        return deepcopy(self)

def amplifier(x_in, y_in, NF, dB_gain, f_gain = 0, Zin = 50, Zout = 50, time = True): # return y_in with gain applied
    # TODO: this function is missing lots of detail and will need to be heavily rewritten to integrate with other code

    print(len(f_gain))

    # TODO: input mismatch, confirm Vf2Pf works for f-dependent Z0
    if time:
        t = x_in
        f = fft_fs(x_in)
        Pf_in = Vt2Pf(y_in, len(x_in))
    else:
        t = x_in
        f = x_in
        Pf_in = y_in

    # TODO: f-dependent gain
    Pf_out = Pf_in * dB_gain
    Pf_out = Pf_out + Vt_noise()

    # TODO: output mismatch, confirm Pf2Vf works for f-dependent Z0
    if time:
        return Pf2Vt(Pf_out, len(x_in))
    else:
        return Pf_out


# Useful time-domain voltages

def make_time(ns, t_max):
    return np.linspace(0, t_max, ns)

def V_psk(t_sample, fc, f_sym, data, dBm, n = 1): # create (2**n)-PSK modulated signal (circular constellation), with carrier fc and symbol rate f_sym
    # expected data format: "0100100101..." (spaces permitted for readability, will be ignored)

    syms = data2sym(data, n)

    m = 2**n
    d_phi = 2*pi/m # get phase step
    T_sym = 1/f_sym # get symbol time

    phi = d_phi * np.array([syms[int(t/T_sym)] - 0.5*np.sign(syms[int(t/T_sym)])for t in t_sample])

    return dBm2Vp(dBm) * np.sin((f2w(fc) * t_sample) + phi)

def V_fsk(t_sample, fc, f_sym, f_dev, data, dBm, n = 1): # create (2**n)-FSK modulated signal, with carrier fc, symbol rate f_sym, deviation f_dev
    # expected data format: "0100100101..." (spaces permitted for readability, will be ignored)

    syms = data2sym(data, n)
    T_sym = 1/f_sym # get symbol time

    f = fc + f_dev * np.array([syms[int(t/T_sym)] for t in t_sample])

    return dBm2Vp(dBm) * np.sin((f2w(f) * t_sample))

def V_msk(t_sample, fc, f_sym, data, dBm): # create MSK modulated signal, n = 1, m = 2
    # expected data format: "0100100101..." (spaces permitted for readability, will be ignored)
    # see: https://www.dsprelated.com/showarticle/1016.php
    # see https://www.slideshare.net/mahinthjoe/lecture-5-1580898
    syms = data2sym(data)
    T_sym = 1/f_sym # get bit time
    h = 0.5
    f_dev = h/(2*T_sym)

    odd_bits = [syms[2*i] for i in range(int(math.ceil(len(syms)/2)))]
    even_bits = [-1] + [syms[2*i + 1] for i in range(len(syms)//2)]

    inverted = np.array([odd_bits[int(t/(2*T_sym))] for t in t_sample])
    delta = np.array([(abs(odd_bits[int(t/(2*T_sym))] + even_bits[int((t+T_sym)/(2*T_sym))]) - 1) for t in t_sample])

    return dBm2Vp(dBm) * (inverted * delta * -1) * np.sin((f2w(fc + (delta * f_dev)) * t_sample))
    # Extremely verbose debug return
    # return (dBm2Vp(dBm) * (inverted * delta * -1) * np.sin((f2w(fc + (delta * f_dev)) * t_sample)), dBm2Vp(dBm) * inverted, dBm2Vp(dBm) * delta, np.array([dBm2Vp(dBm) * odd_bits[int(t/(2*T_sym))] for t in t_sample]), np.array([dBm2Vp(dBm) * even_bits[int((t+T_sym)/(2*T_sym))] for t in t_sample]), np.array([dBm2Vp(dBm) * data[int(t/T_sym)] for t in t_sample]))

def data2sym(data, n = 1): # convert string of 1's and 0's to symbols format
    # output symbols format: [x1, x2, ... xm], -m/2 <= xi <= m/2, m != 0, m = 2**n

    bs = "".join(data.split()) # remove internal spaces
    if len(bs) % n != 0: return 'fail'

    k = 2**(n-1) # number of different states per symbol

    return [int(bs[i:i+n],2) - int(k) + int(bs[i]) for i in range(0, len(bs), n)]

def sym2data(sym, n = 1):
    k = 2**(n-1) # number of different states per symbol
    if max(np.abs(sym)) > k: return fail

    return " ".join(["{:0{}b}".format(s+k-(1 if s>0 else 0), n) for s in sym])

# Network voltages

# @np.vectorize
def Vdiv_proto(Z1, Z2):
    if Z1 == inf or Z2 == 0: return 0
    if Z2 == inf: return 1

    return Z2/(Z1+Z2)

Vdiv = np.vectorize(Vdiv_proto, otypes = [np.complex])

def Pdiv_proto(Z1, Z2):
    rtH = Vdiv_proto(Z1, Z2)
    return rtH * mag(rtH)

Pdiv = np.vectorize(Pdiv_proto, otypes = [np.complex])

def Znetwork(series, shunt):
    if np.shape(series) != np.shape(shunt): return "fail"

    v = np.zeros(np.shape(series), dtype=np.complex)
    z_shunt_eq = np.zeros(np.shape(series), dtype=np.complex)

    for antinode in range(1,len(series)+1):
        node = len(series)-antinode # this is dumb

        if antinode == 1:
            z_shunt_eq[node] = shunt[node]
        else:
            z_shunt_eq[node] = par(shunt[node],series[node+1] + z_shunt_eq[node+1])

    for node in range(0,len(series)):
        if node == 0:
            v[node] = Vdiv(series[node],z_shunt_eq[node])
        else:
            v[node] = v[node-1] * Vdiv(series[node],z_shunt_eq[node])

    return [v,z_shunt_eq]

# RF propagation

def fspl(d, w, dB = True):
    if dB: return dBv(2.0*w*d/c)
    return (2.0*w*d/c)**2.0

# plotting

@ticker.FuncFormatter
def HzFormatter(x, pos):
    if abs(x) >= 1e12:
        hz = 'THz'
        f = x/1.0e6
    elif abs(x) >= 1e9:
        hz = 'GHz'
        f = x/1.0e9
    elif abs(x) >= 1e6:
        hz = 'MHz'
        f = x/1.0e6
    elif abs(x) >= 1e3:
        hz = 'kHz'
        f = x/1.0e3
    else:
        hz = 'Hz'
        f = x
    return "{:.1f} {}".format(f, hz)

def plot_power_spectrum(ax, x, y, time = False, Z0 = 50, **kwargs):
    if not time:
        fs = x
        Pf = y
        ax.plot(fs, W2dBm(mag(Pf)), **kwargs)
    if time:
        fs = fft_fs(x) # x = ts
        Pf = Vt2Pf(y, len(x), Z0)
        ax.plot(fs, W2dBm(mag(Pf)), **kwargs)

    ax.xaxis.set_major_formatter(HzFormatter)

# Gold code generation

def gold_codes(m):
    # Generates 2^m + 1 Gold codes, each of length 2^m - 1
    # Valid for m % 4 != 0, practical for m < 16
    # see https://web.archive.org/web/2 0070112230234/http://paginas.fe.up.pt/~hmiranda/cm/Pseudo_Noise_Sequences.pdf page 14
    # See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.max_len_seq.html#scipy.signal.max_len_seq

    if m % 4 == 0 or m >= 16: return 'fail - invalid m'

    N = 2**m - 1

    reg = np.zeros(m)
    reg[0] = 1 # to create nonzero initial conditions

    mls1 = np.zeros(N)

    if m == 3:
        prim_poly = np.array([1, 1, 0]) # ignore degree-m coefficient
    if m == 5:
        prim_poly = np.array([1, 0, 1, 0, 0])
    if m == 6:
        prim_poly = np.array([1, 1, 0, 0, 0, 0])
    if m == 7:
        prim_poly = np.array([1, 1, 0, 0, 0, 0, 0])
    if m == 9:
        prim_poly = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])
    if m == 10:
        prim_poly = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    if m == 11:
        prim_poly = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    if m == 13:
        prim_poly = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    if m == 14:
        prim_poly = np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    if m == 15:
        prim_poly = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # larger values of m take a _very_ long time to generate and consume significant memory and are not recommended
    #
    # if m == 17:
    #     prim_poly = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # if m == 18:
    #     prim_poly = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # if m == 19:
    #     prim_poly = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # if m == 21:
    #     prim_poly = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # if m == 22:
    #     prim_poly = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # if m == 23:
    #     prim_poly = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(N):
        mls1[i] = reg[0]
        fb = sum(prim_poly * reg) % 2
        # print('{} {} {}'.format(reg, out[x], fb))
        reg = np.roll(reg, -1)
        reg[-1] = fb

    # see p. 13
    if m % 4 == 2: k = 2
    else: k = 1

    q = 2**k + 1

    mls2 = [mls1[(i*q) % N] for i in range(N)] # decimate mls1 by Q; take every qth element

    x = 0b0
    y = 0b0

    for i in range(N):
        x = (x<<1) | int(mls1[i])
        y = (y<<1) | int(mls2[i])

    # print('{:07b} {:07b}'.format(x,y))

    return [x ^ ( ((y >> m)|(y << N-m)) & (2**N - 1) ) for m in range(N)] + [x, y]

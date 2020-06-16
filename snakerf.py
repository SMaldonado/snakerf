import math
from math import e, pi, inf
import numpy as np
from numpy import log2, log, log10, rad2deg, deg2rad, sqrt
from numpy.linalg import det
from scipy import interpolate
import matplotlib.ticker as ticker
from copy import deepcopy

c = 299792458.0 # speed of light in vacuum, m/s
kB = 1.380649E-23 # Boltzmann constant, J/K
t0 = 290 # noise figure temperature, Kelvin
e0 = 8.854187e-12 # vacuum permittivity
mu0 = 1.256637e-6 # vacuum permeability
rho_cu = 1.72e-8 # copper resistivity, ohm*m

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

def rms(x):
    return np.std(x)

# passive component frequency responses

def safe_reciprocal_proto(x):
    if x == 0: return inf
    if x == inf: return 0
    return 1/x

Z2Y = np.vectorize(safe_reciprocal_proto, otypes = [np.complex])
Y2Z = np.vectorize(safe_reciprocal_proto, otypes = [np.complex]) # yes, they're the same

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

def undB(dB): # dB power gain to linear power gain
    return 10.0**(dB/10.0)

def dBv(x): # linear voltage gain to dB gain
    return 20*log10(mag(x))

def undBv(dB): # dB gain to linear voltage gain
    return 10.0**(dB/20.0)

def dB2Np(dB): # dB gain to Neper gain
    return dB/dBV(e)

def Np2dB(dB): # Neper gain to dB gian
    return dB*dBV(e)

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

def m2in(m):
    return m * 39.3701

def in2m(inch):
    return inch / 39.3701

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
    # see also: https://community.sw.siemens.com/s/article/what-is-a-power-spectral-density-psd
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

def Vt_thermal_noise(ts, fs, T_noise = t0, Z0 = 50, out_Vf = False): # create sampled time-domain additive white Gaussian voltage noise of specified noise temperature
    # see: https://www.ti.com/lit/an/slva043b/slva043b.pdf
    # see: https://en.wikipedia.org/wiki/Noise_temperature
    # see: https://www.ietlabs.com/pdf/GR_Appnote/IN-103%20Useful%20Formulas,%20Tables%20&%20Curves%20for.pdf
    # see: https://en.wikipedia.org/wiki/Noise_spectral_density
    # see: https://training.ti.com/system/files/docs/1312%20-%20Noise%202%20-%20slides.pdf
    # see: https://electronics.stackexchange.com/questions/303337/fourier-transform-of-additive-white-gaussian-noise
    # see: https://www.gaussianwaves.com/2013/11/simulation-and-analysis-of-white-noise-in-matlab/   --- explains non-flat spectrum
    # tl;dr: appropriately mathematically representing white noise is hard. Buyer beware.

    if T_noise == 0:
        if out_Vf:
            return np.zeros(len(fs))
        else:
            return np.zeros(len(ts))

    # Noise voltage variance (usually ̅V^2) as a single-sided spectral density usually equals:
    # ̅V^2/B = 4*kB*T*R
    V2_noise_Hz = 4 * kB*T_noise*mag(Z0)

    # fill full sampling bandwidth of t_sample with white noise - note that this may not always be desired
    f_nyq = max(fs)
    V_stddev_noise = sqrt(V2_noise_Hz * f_nyq)
    noise = np.random.normal(0, V_stddev_noise, len(ts))

    if out_Vf: return Vt2Vf(noise, len(ts))
    else: return noise

def Vt_background_noise(ts, fs, Z0 = 50, out_Vf = False):
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

    df = fs[1]-fs[0]

    Pf_noise_white = Vt2Pf(V_noise_white, len(ts))
    V2_Hz_mean_noise_white = np.mean(mag(Pf_noise_white * Z0 / df))
    H_norm = V2_noise_Hz / (V2_Hz_mean_noise_white)
    Pf_noise = Pf_noise_white * H_norm

    # Total integrated power agrees decently well between PSD and output power spectrum
    # V2_var_noise_out = np.trapz(mag(Pf_noise * Z0 / df), fs) # integrate total noise power
    # print(V2_var_noise)
    # print(V2_var_noise_out)

    if out_Vf: return Pf2Vf(Pf_noise, Z0)
    else: return Pf2Vt(Vf_noise, len(ts), Z0)

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
        self.df = self.fs[1] - self.fs[0]

        if sig == 0: # initialize to zero signal
            self.Vt = np.zeros(len(self.ts))
            self.Pf = np.zeros(len(self.fs))
            self.Vf = np.zeros(len(self.fs))
        else:
            if sig_Vt: # initialize to provided Vt
                self.update_Vt(sig)
            else: # initialize to provided Vf
                self.update_Vf(sig)

    def update_Vt(self, Vt): # update time-domain voltage and ensure f-domain consistency
        if len(Vt) != len(self.ts): IndexError('signal and sample times have different lengths')
        self.Vt = Vt
        self.Vf = Vt2Vf(self.Vt, self.ns)
        self.Pf = Vt2Pf(self.Vt, self.ns, self.Z0)

    def update_Vf(self, Vf): # update power spectrum and ensure time-domain consistency
        if len(Vf) != len(self.fs): IndexError('signal and frequency have different lengths')
        self.Vf = Vf
        self.Pf = Vf2Pf(self.Vf, self.ns, self.Z0)
        self.Vt = Vf2Vt(self.Vf, self.ns)

    def make_tone(self, f, P_dBm):
        self.update_Vt(dBm2Vp(P_dBm) * np.sin(f2w(f) * self.ts))

    def make_square(self, f, AV):
        self.update_Vt(AV * (-1)**np.floor(2*self.ts*f))

    def add_noise(self, noise = t0, NF = False):
        if NF: T_noise = NF2T_noise(noise)
        else: T_noise = noise

        # EXPERIMENTAL: Add noise by directly summing voltages, not power combining
        # see https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
        self.update_Vt(self.Vt + Vt_thermal_noise(self.ts, self.fs, T_noise = T_noise, Z0 = self.Z0))

    def amplify(self, amp):
        pass # TODO: fix for new two-port
        # TODO: input mismatch, confirm Vf2Pf works for f-dependent Z0
        # self.add_noise(noise = amp.NF, NF = True)

        # self.update_Pf(self.Pf * undB(amp.gain(self.fs)))

        # TODO: output mismatch, confirm Pf2Vf works for f-dependent Z0

    def copy(self):
        return deepcopy(self)

    def plot_f(self, ax, **kwargs):
        plot_power_spectrum(ax, self.fs, self.Pf, False, self.Z0, **kwargs)


def _make_b_ser(Zser):
    return np.array([[[1, -Z],[0, 1]] for Z in Zser])

def _make_b_shunt(Zshunt):
    return np.array([[[1, 0],[-Z2Y(Z), 1]] for Z in Zshunt])

def _make_b_tl(Z0, gamma, l):
    return np.array([[[np.cosh(y*l), Z*np.sinh(y*l)],[-Z2Y(Z)*np.sinh(y*l), np.cosh(y*l)]] for Z, y in zip(Z0, gamma)])

# see https://en.wikipedia.org/wiki/Two-port_network#Interrelation_of_parameters
# see https://en.wikipedia.org/wiki/Two-port_network#Table_of_transmission_parameters
class Two_Port: # Represents a noisy 2-port object with gain
    def __init__(self, fs, b, NF = 0):
        self.fs = fs
        self.b = b
        self.NF = NF

        # TODO: handle f-dependent noise

    def Z_in(self, Z_term = None, port = 1):

        if Z_term is None:
            Zl = _make_b_shunt(Zopen(f2w(self.fs)))
        elif len(Z_term) != len(self.fs):
            raise IndexError('termination impedance and frequency have different lengths')
        else:
            Zl = _make_b_shunt(Z_term)

        b = self.b

        if port == 1:
            b = Zl @ b
            return -b[:, 1, 1]/b[:, 1, 0]
        elif port == 2:
            b = b @ Zl
            return -b[:, 0, 0]/b[:, 1, 0]
        else:
            return 'fail' # TODO: real exception

        # return np.array( (b[:, 0, 1] - Zl*b[:, 1, 1]) / (Zl*b[:, 1, 0] - b[:, 0, 0]) ) # this works but not for Zl == inf

    def V_out(self, Zs, Zl):
        if len(Zs) != len(self.fs) or len(Zl) != len(self.fs): raise IndexError('impedances and frequencies are not all same length')

        b = self.b

        Z_in = self.Z_in(Zl)

        # print(Zin)

        V1 = Vdiv(Zs, Z_in)
        V2 = V1 * det(b) / (b[:, 1, 1] - (b[:, 0, 1]/Zl))

        # V2 = V1 * (b[:, 0, 0] - b[:, 0, 1]*b[:, 1, 0]) / (b[:, 1, 1] - (b[:, 0, 1]/Zl))

        return V2

    @classmethod
    def from_network(cls, fs, series, shunt, NF_dB = 0):
        if abs(len(series) - len(shunt)) > 1: raise IndexError('series and shunt have different lengths')

        j = len(series) - len(shunt) # offset between series and shunt elements

        if j == -1: # shunt is longer than series; shunt first
            b = _make_b_shunt(shunt[0])
        else:
            b = np.array([np.identity(2) for f in fs])

        for i in range(min(len(series), len(shunt))):
            b_ser = _make_b_ser(series[i])
            b_shunt = _make_b_shunt(shunt[i - min(j,0)])

            # -- [ b ] -- [ser] -- | -- ...
            #                   [shunt]
            #                      V

            b = b_shunt @ b_ser @ b

        if j == 1:
            b = _make_b_ser(series[-1]) @ b

        # TODO: Calculate f-dependent noise

        return cls(fs, b, NF_dB)

    @classmethod
    def from_tl(cls, fs, RLGC, l): # initialize from primary line constants (per length), which are either contant or functions of frequency
        # unit length can be any unit so long as consistent between all parameters

        # if not (len(RLGC) == 1 or len(RLGC) == len(fs)): return 'fail' # TODO: real exception

        R = RLGC[0]
        L = RLGC[1]
        G = RLGC[2]
        C = RLGC[3]

        ws = f2w(fs)
        Z = R + 1j*ws*L # these conveniently work for either len(R) == 1 or len(R) == len(ws)
        Y = G + 1j*ws*C

        Z0 = sqrt(Z/Y)
        gamma = sqrt(Z*Y)

        return cls(fs, _make_b_tl(Z0, gamma, l))


    @classmethod
    def from_gain(cls, fs, dB_gain, NF_dB = 0, f_gain = None, Zin = 50, Zout = 50):
        if f_gain == None:
            gain = dB_gain * np.ones(len(fs))

        H = interpolate.interp1d(log10(np.array(f_gain)), dB_gain, fill_value="extrapolate")

        safe_fs = fs
        safe_fs[safe_fs <= 0] = 0.1 # TODO: make more rigorous
        gain = H(log10(safe_fs))

        # TODO: Calculate b
        # TODO: Calculate f-dependent noise

        # return cls(fs, b, NF_dB)


def RLGC_from_microstrip(fs, Dk, Df, R_ins, h, w, t = 0.0014):
    # see http://web.mst.edu/~marinak/files/my_publications/papers/Causal_RLGC.pdf
    # see https://technick.net/tools/impedance-calculator/microstrip/

    ws = f2w(fs)

    e_i = Dk*e0
    e_ii = Df*e_i # Stanford EE 273 Lecture 4 Slide 16
    er = Dk # TODO: fix (this is an oversimplification)

    R0 = m2in(rho_cu) / (w * t)
    Rs = sqrt(pi*mu0*rho_cu)/w # Stanford EE 273 Lecture 4 Slide 10

    G0 = Z2Y(R_ins)

    Z0 = (87/sqrt(er + 1.41)) * log(5.98*h/(0.8*w + t)) # = sqrt(L0 / C0)
    td_l = 85e-12 * sqrt(0.475 * er + 0.67) # = sqrt(L0 * C0)

    C0 = td_l / Z0
    L0 = td_l * Z0

    R = R0 + sqrt(fs)*Rs
    L = L0 + Rs/(2*pi*sqrt(fs))
    G = G0 + 2*pi*fs*C0*Df
    C = C0 # Kg*e_i

    return np.array([R, L, G, C])

    # Johnson, H. W. and Graham, M., “High Speed Digital Design – A Handbook of Black Magic”, Prentice Hall, 1993, pp 187
    # all dimensions in inches, valid for 0.1 < w/h < 2.0, er < 15
    # microstrip:
    # Z0 = (87/sqrt(er + 1.41)) * ln(5.98*h/(0.8*w + t))
    # td_l = 85 * sqrt(0.475 * er + 0.67)

def RLGC_from_stripline(fs, Dk, Df, R_ins, h, w, t = 0.0014):
    # see http://web.mst.edu/~marinak/files/my_publications/papers/Causal_RLGC.pdf
    # see https://technick.net/tools/impedance-calculator/microstrip/

    ws = f2w(fs)

    e_i = Dk*e0
    e_ii = Df*e_i # Stanford EE 273 Lecture 4 Slide 16
    er = Dk # TODO: fix (this is an oversimplification)

    R0 = m2in(rho_cu) / (w * t)
    Rs = sqrt(pi*mu0*rho_cu)/w # Stanford EE 273 Lecture 4 Slide 10

    G0 = Z2Y(R_ins)

    Z0 = (60/sqrt(er)) * log(3.8*h / (0.8*w + t)) # = sqrt(L0 / C0)
    td_l = 85e-12 * sqrt(er) # = sqrt(L0 * C0)

    C0 = td_l / Z0
    L0 = td_l * Z0

    R = R0 + sqrt(fs)*Rs
    L = L0 + Rs/(2*pi*sqrt(fs))
    G = G0 + 2*pi*fs*C0*Df
    C = C0 # Kg*e_i

    return np.array([R, L, G, C])

    # Johnson, H. W. and Graham, M., “High Speed Digital Design – A Handbook of Black Magic”, Prentice Hall, 1993, pp 188
    # all dimensions in inches, valid for w/2h < 0.35, t/2h < 0.25, er < 15
    # stripline:
    # Z0 = (60/sqrt(er)) * ln(3.8*h / (0.8*w + t))
    # td_l = 85 * sqrt(er)

# TODO: Port impedances/mismatch, nonlinearities, noise
class Mixer:
    def __init__(self, Z_port = [50,50,50], gain_dB = 0, NF_dB = 0):
        self.Z_LO = Z_port[0]
        self.Z_IF = Z_port[1]
        self.Z_RF = Z_port[2]

    def mix(self, sig_f, sig_lo, Zs = 50):
        out = sig_f.copy()
        out.update_Vt(sig_f.Vt * sig_lo.Vt / rms(sig_lo.Vt))

        return out

# Modulation and demodulation

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
    if len(bs) % n != 0: raise ValueError('data length not divisible by n')

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

# see https://en.wikipedia.org/wiki/Two-port_network#Collapsing_a_two-port_to_a_one_port
def Znetwork(series, shunt):
    if np.shape(series) != np.shape(shunt): raise IndexError('series and shunt have different lengths')

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

def Gamma_proto(Z0, ZL): # reflection coefficient
    if Z0 == inf or ZL + Z0 == 0: raise ValueError('invalid Z0')
    if ZL == inf: return 1

    return (ZL - Z0) / (ZL + Z0)

Gamma = np.vectorize(Gamma_proto, otypes = [np.complex])

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
    return "{:.2f} {}".format(f, hz)

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

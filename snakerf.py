import math
from math import pi, inf
import numpy as np
from numpy import log10, rad2deg, deg2rad, sqrt
import matplotlib.ticker as ticker

c = 299792458.0 # speed of light in vacuum, m/s
kB = 1.380649E-23 # Boltzmann constant, J/K

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

def C(C, w, ESR = 0):
    if C == inf: return np.zeros(len(w)) + ESR
    if C == 0: return np.ones(len(w)) + inf
    return (1/(1j*w*C)) + ESR

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

# TODO: convert everything to rfft to save space and to agree better with voltage noise PSDs

def fft_fs(t_sample):
    timestep = (max(t_sample) - min(t_sample))/len(t_sample)
    return np.fft.fftfreq(len(t_sample), d = timestep)

def Vf2Pf(Vf, Z0 = 50): # f-domain voltage to f-domain power
    return mag(Vf) * (Vf.real + 1j*Vf.imag) / (2*Z0)

def Pf2Vf(Pf, Z0 = 50): # f-domain power to f-domain voltage
    magnitude = sqrt(mag(Pf) * 2*Z0)
    phase = Pf / mag(Pf)

    return magnitude * (phase.real + 1j*phase.imag)

def Vt2Vf(Vt, ns): # time-domain voltage to f-domain voltage
    # see https://www.sjsu.edu/people/burford.furman/docs/me120/FFT_tutorial_NI.pdf
    return np.fft.fft(Vt)* 2/ns # numpy fft scaling; see https://numpy.org/doc/stable/reference/routines.fft.html#normalization

def Vf2Vt(Vf, ns): # f-domain voltage to time-domain voltage
    return np.fft.ifft(Vf * ns/2).real # numpy fft scaling; see https://numpy.org/doc/stable/reference/routines.fft.html#normalization

def Vt2Pf(Vt, ns, Z0 = 50): # time-domain voltage to f-domain power
    return Vf2Pf(Vt2Vf(Vt, ns), Z0)

def Pf2Vt(Pf, ns, Z0 = 50): # f-domain power to time-domain voltage
    return Vf2Vt(Pf2Vf(Pf, Z0), ns)

def Vt_noise(t_sample, T_noise, Z0 = 50): # create sampled time-domain additive white Gaussian voltage noise of specified noise temperature
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
    f_sample = len(t_sample)/(max(t_sample) - min(t_sample))
    # print(f_sample)
    V_stddev_noise = sqrt(V2_noise_Hz * f_sample / 2)
    # print(V_stddev_noise * 3)
    return np.random.normal(0, V_stddev_noise, len(t_sample))

def power_combine(Vts, ts, Z0 = 50, out_Pf = False): # power combine array of time-domain voltages Vts
    ns = len(ts)
    Pf = np.sum([Vt2Pf(Vt, ns) for Vt in Vts], axis = 0) # sum elementwise

    if out_Pf: return Pf
    return Pf2Vt(Pf, ns, Z0)

# Network voltages

@np.vectorize
def Vdiv(Z1, Z2):
    if Z2 == 0: return 0
    return Z2/(Z1+Z2)

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

def plot_power_spectrum(ax, x, y, time = False, Z0 = 50):
    if not time:
        fs = x
        Pf = y
        ax.plot(np.fft.fftshift(fs), np.fft.fftshift(W2dBm(mag(Pf))))
    if time:
        fs = fft_fs(x) # x = ts
        Pf = Vt2Pf(y, len(x), Z0)
        ax.plot(np.fft.fftshift(fs), np.fft.fftshift(W2dBm(mag(Pf))))

    ax.xaxis.set_major_formatter(HzFormatter)

# Gold code generation

def gold_codes(m):
    # Generates 2^m + 1 Gold codes, each of length 2^m - 1
    # Valid for m % 4 != 0, practical for m < 16
    # see https://web.archive.org/web/2 0070112230234/http://paginas.fe.up.pt/~hmiranda/cm/Pseudo_Noise_Sequences.pdf page 14

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

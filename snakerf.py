import math
from math import pi, inf
import numpy as np
from numpy import log10, rad2deg, deg2rad, sqrt
import matplotlib.ticker as ticker

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
    return 20*log10(abs(x))

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

# matplotlib x-axis labeling

@ticker.FuncFormatter
def HzFormatter(x, pos):
    if x >= 1e12:
        hz = 'THz'
        f = x/1.0e6
    elif x >= 1e9:
        hz = 'GHz'
        f = x/1.0e9
    elif x >= 1e6:
        hz = 'MHz'
        f = x/1.0e6
    elif x >= 1e3:
        hz = 'kHz'
        f = x/1.0e3
    else:
        hz = 'Hz'
        f = x
    return "{:.1f} {}".format(f, hz)

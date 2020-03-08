import math
from math import pi
import numpy as np
from numpy import log10, rad2deg, deg2rad

def par(Z1, Z2):
    return (Z1 * Z2)/(Z1 + Z2)

def ser(Z1, Z2):
    return Z1 + Z2

def phase(x):
    return np.angle(x)

# passive component frequency responses

def C(C, w, ESR = 0):
    return (1/(1j*w*C)) + ESR

def L(L, w, ESR = 0):
    return 1j*w*L + ESR

def R(R, w):
    return R * np.ones(len(w))

# dB handling

def dB(x):
    return 10*log10(x)

def undB(x):
    return 10.0**(x/10.0)

def dBv(x):
    return 20*log10(abs(x))

# Unit conversions

def W2dBm(W):
    return dB(W) + 30

def dBm2W(dBm):
    return undB(dBm - 30.0)

def Vp2dBm(Vp,  Z0 = 50):
    return W2dBm(Vp**2.0 / (2.0*Z0))

# Network voltages

def Vdiv(Z1, Z2):
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

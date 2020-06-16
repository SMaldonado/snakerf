import snakerf as srf
import numpy as np


bs = '000 001 010 011 100 101 110 111'
sym = srf.data2sym(bs,n = 3)
bs2 = srf.sym2data(sym, n = 3)
print(bs)
print(sym)
print(bs2)

import numpy as np
import camb
from matplotlib import pyplot as plt


planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)

print(planck.shape)
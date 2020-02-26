'''
global_settings.py

a script to run in every notebook to set consistent, global settings

Jorge Ramirez
July 23, 2019
'''

import numpy as np
import scipy.constants as scc
import scipy.special as special

import matplotlib.pyplot as plt

import os
from matplotlib.backends.backend_pdf import PdfPages       #For saving figures to single pdf

import scipy.signal as sig
import scipy.io as sio

import time
import timeit

''' global plotting settings '''
#plt.style.use('seaborn-paper')
# Update the matplotlib configuration parameters:
plt.rcParams.update({'text.usetex': False,
                     'lines.linewidth': 3,
                     'font.family': 'sans-serif',
                     'font.serif': 'Helvetica',
                     'font.size': 14,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'axes.labelsize': 'large',
                     'axes.titlesize': 'large',
                     'axes.grid': True,
                     'grid.alpha': 0.53,
                     'lines.markersize': 12,
                     'legend.borderpad': 0.2,
                     'legend.fancybox': True,
                     'legend.fontsize': 'medium',
                     'legend.framealpha': 0.7,
                     'legend.handletextpad': 0.1,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'figure.figsize': (12,8),
                     'savefig.dpi': 100,
                     'pdf.compression': 9})


plotsavelocation = "plots/{}"  # save plots in the "plots" folder
datasavelocation = "data/{}"   # save data in "data" folder


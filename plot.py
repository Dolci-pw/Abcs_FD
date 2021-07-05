#==============================================================================
# Python Imports
#==============================================================================
import matplotlib
# matplotlib.use('Agg')
import numpy                  as np
import matplotlib.pyplot      as plot
import matplotlib.ticker      as mticker  
from matplotlib               import cm
from mpl_toolkits.axes_grid1  import make_axes_locatable
from matplotlib               import ticker
from   numpy             import linalg              as la
import sys
#==============================================================================
from   devito           import *
import settings_config
sys.path.insert(0, './code')
from   timeit import default_timer as timer
import solver, domain2D, utils, velmodel
plot.rc('text', usetex=True)

def dif(abc, true, pn, pref):
    d  = abc[pn:-pn, 0:-pn] - true[pref:-pref, 0:-pref]
    n2 = la.norm(np.reshape(d[:],-1),2)/la.norm(np.reshape(true[pref:-pref, 0:-pref],-1),2)
    return n2

setting = settings_config.settings.setting5
setup   = utils.ProblemSetup(setting)
n2 = 0
n2_noabc  = []
n2_cpml   = []
n2_pml    = []
n2_higdon = []

sd     = setting["shots_dist"]
nshots = int((setting["lenx"]-200)/sd)+1
for sn in range(0, nshots):

    true       = np.load("data_save/rec5_ref_sigma10" + str(sn) + ".npy")
    noabc      = np.load("abc_analysis/rec5_noabc_sigma10" + str(sn) + ".npy")
    abc_cpml   = np.load("data_save/rec5_cpml_sigma10" + str(sn) + ".npy")
    abc_pml    = np.load("data_save/rec5_pml_sigma10" + str(sn) + ".npy")
    abc_higdon = np.load("data_save/rec5_higdon_sigma10" + str(sn) + ".npy")

    pref = 500 # referency field has 500 points in the extended region
    pn   = 0
    n2   = dif(noabc, true, pn, pref)
    n2_noabc.append(n2)

    pn = 30   # Points number in the extended region for cpml method
    n2 = dif(abc_cpml, true, pn, pref)
    n2_cpml.append(n2)

    pn = 70   # Points number in the extended region for pml method
    n2 = dif(abc_pml, true, pml, pref)
    n2_pml.append(n2)

    pn = 40   # Points number in the extended region for Higdon method
    n2 = dif(abc_higdon, true, 40, pref)
    n2_higdon.append(n2)


x = np.linspace(setup.x0pml+100,setup.x1pml-100,nshots)

plot.plot(x, n2_cpml, 'o--m', mec='m', ms=9, mew=3,label=r'CPML')
plot.plot(x, n2_pml, 'p--g', mec='g', ms=9, mew=3,label=r'PML')
plot.plot(x, n2_higdon, 'd--r', mec='r', ms=9, mew=3,label=r'HABC-Higdon')
plot.ylabel(r'\Huge{$d_{rec}$}', fontsize=24)
plot.xlabel(r'\Huge{$x$}', fontsize=24)
plot.legend(loc='upper right')
plot.tick_params(labelsize=24)
plot.show()



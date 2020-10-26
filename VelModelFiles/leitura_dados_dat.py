#==============================================================================
# Python Imports
#==============================================================================
import numpy             as np
import math              as mt
import time              as tm
import sys
import segyio
import matplotlib.pyplot               as plot
import matplotlib.ticker      as mticker  
from matplotlib               import cm
from mpl_toolkits.axes_grid1  import make_axes_locatable
from matplotlib               import ticker
#==============================================================================

#==============================================================================
plot.close("all")
#==============================================================================

#==============================================================================
# Data Parameters  
#==============================================================================
nptx = 1407
nptz = 311

x0 = 0
x1 = 70300

z0 = 0
z1 = 9920

X = np.linspace(x0,x1,nptx)
Z = np.linspace(z0,z1,nptz)

dados = [x0,x1,nptx,z0,z1,nptz]
#==============================================================================

#==============================================================================
# Data Reading  
#==============================================================================
vel            = np.loadtxt('gm_perfil2.dat')
vel            = np.delete(vel,0,1)
fscale         = 10**(-3) 
vel            = vel*fscale
np.save('gm_perfil_dat',vel)
#==============================================================================

#==============================================================================
# Plot Velocity
#==============================================================================
def graph2dvel(vel,dados):
    
    x0   = dados[0]
    x1   = dados[1]
    nptx = dados[2]
    z0   = dados[3]
    z1   = dados[4]
    nptz = dados[5]
    
    plot.figure(figsize = (14,10))
    
    fscale =  10**(-3)
    
    vminv = np.amin(vel)

    vmaxv = np.amax(vel)
    
    scale  = np.amax(vel)

    extent = [fscale*x0,fscale*x1, fscale*z1, fscale*z0]

    fig = plot.imshow(np.transpose(vel), vmin=vminv,vmax=vmaxv, cmap=cm.jet, extent=extent)
        
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))

    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))

    plot.title('Velocity Profile')

    plot.grid()

    ax = plot.gca()

    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
    
    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
   
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="4%", pad=0.05)

    tick_locator = ticker.MaxNLocator(nbins=5)

    cbar = plot.colorbar(fig, cax=cax, format='%.2e')

    cbar.locator = tick_locator

    cbar.update_ticks()

    cbar.set_label('Velocity [km/s]')

    ttl = ax.title
    
    ttl.set_position([.5, 1.1])

    plot.show()
    
    #plot.close()

    return
#==============================================================================

#==============================================================================
# Plots
#==============================================================================
P1 = graph2dvel(vel,dados)
#==============================================================================
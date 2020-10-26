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
xstart = 0
xend   = 1407

zstart = 0
zend   = 311

deltax = 50.0
deltaz = 32.0

nptx = xend - xstart
nptz = zend - zstart

x0 = 0
x1 = (nptx-1)*deltax

z0 = 0
z1 = (nptz-1)*deltaz

X = np.linspace(x0,x1,nptx)
Z = np.linspace(z0,z1,nptz)

dados = [x0,x1,nptx,z0,z1,nptz]
#==============================================================================

#==============================================================================
# Data Reading  
#==============================================================================
with segyio.open('modelC3.segy',ignore_geometry = True) as segyfile:
    #vel = segyio.tools.cube(segyfile)[0,xstart:xend,zstart:zend]
    
    #     # Memory map file for faster reading (especially if file is big...)
    segyfile.mmap()

    #     # Print binary header info
    print(segyfile.bin)
    print(segyfile.bin[segyio.BinField.Traces])

    #     # Read headerword inline for trace 10
    print(segyfile.header[11][segyio.TraceField.INLINE_3D])

    #     # Print inline and crossline axis
    print(segyfile.xlines)
    print(segyfile.ilines)
    
#fscale = 10**(-3) 
#vel    = vel*fscale
#np.save("gm_perfilnovo_segy",vel)
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
#P1 = graph2dvel(vel,dados)
#==============================================================================
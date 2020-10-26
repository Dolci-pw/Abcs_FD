#==============================================================================
# Python Imports
#==============================================================================
import numpy                  as np
import math                   as mt
import time                   as tm
import sys
import segyio
import matplotlib.pyplot      as plot
import matplotlib.ticker      as mticker  
from matplotlib               import cm
from mpl_toolkits.axes_grid1  import make_axes_locatable
from matplotlib               import ticker
#==============================================================================

#==============================================================================
plot.close("all")
#==============================================================================

#==============================================================================
# Data Reading  
#==============================================================================
vel1 = np.load("GM_2019_CL7746_(z_step_32).npy")
vel2 = np.load("GM_2019_IL1976_(z_step_32).npy")
vel3 = np.load("GM_B5_CL7746_(z_step_25).npy")
vel4 = np.load("GM_B5_IL1976_(z_step_25).npy")
vel5 = np.load("GM_C3_CL7746_(z_step_32).npy")
vel6 = np.load("GM_C3_IL1976_(z_step_32).npy")
vel7 = np.load("gm_perfil_bin.npy")
vel8 = np.load("gm_perfil_dat.npy")
vel9 = np.load("marmousi_perfil_segy.npy")
#==============================================================================

#==============================================================================
# Data Parameters  
#==============================================================================
mvel = 6

if(mvel==1):
    vel = vel1
    deltax = 25.0
    deltaz = 32.0
    
if(mvel==2):
    vel = vel2
    deltax = 25.0
    deltaz = 32.0

if(mvel==3):
    vel = vel3
    deltax = 12.5
    deltaz = 25.0

if(mvel==4):
    vel = vel1
    deltax = 12.5
    deltaz = 25.0

if(mvel==5):
    vel = vel5
    deltax = 12.5
    deltaz = 32.0
    
if(mvel==6):
    vel = vel6
    deltax = 12.5
    deltaz = 32.0

if(mvel==7):
    vel = vel7
    deltax = 50.0
    deltaz = 32.0

if(mvel==8):
    vel = vel8
    deltax = 50.0
    deltaz = 32.0

if(mvel==9):
    vel = vel9
    deltax = 1.25
    deltaz = 1.25

xstart = 0
xend   = vel.shape[0]

zstart = 0
zend   = vel.shape[1]

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
# Plot Velocity
#==============================================================================
def graph2dvel(vel,dados,mvel):
    
    x0   = dados[0]
    x1   = dados[1]
    nptx = dados[2]
    z0   = dados[3]
    z1   = dados[4]
    nptz = dados[5]
    
    plot.figure(figsize = (20,10))
    
    fscale =  10**(-3)
    
    vminv = np.amin(vel)

    vmaxv = np.amax(vel)
    
    scale  = np.amax(vel)

    extent = [fscale*x0,fscale*x1, fscale*z1, fscale*z0]

    fig = plot.imshow(np.transpose(vel), vmin=vminv,vmax=vmaxv, cmap=cm.jet, extent=extent)
        
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))

    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))

    plot.title('Velocity Profile %d'%mvel)

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
    
    ttl.set_position([.5, 1.05])

    plot.savefig('vel_map_%d.png'%mvel)

    plot.show()
    
    #plot.close()

    return
#==============================================================================

#==============================================================================
# Plots
#==============================================================================
P1 = graph2dvel(vel,dados,mvel)
#==============================================================================
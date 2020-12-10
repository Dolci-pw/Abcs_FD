#==============================================================================
# Python Imports
#==============================================================================
import matplotlib
matplotlib.use('Agg')
import numpy                  as np
import matplotlib.pyplot      as plot
import matplotlib.ticker      as mticker  
from matplotlib               import cm
from mpl_toolkits.axes_grid1  import make_axes_locatable
from matplotlib               import ticker
#==============================================================================

#==============================================================================
# Plot 1 - Forward Displacement
#==============================================================================
def graph2d(U,i,setup):
    
    x0pml = setup.x0pml
    x1pml = setup.x1pml
    z0pml = setup.z0
    z1pml = setup.z1pml
    npmlx = setup.npmlx
    npmlz = setup.npmlz
    
    fscale = 10**(-3) 
    
    scale  = np.amax(U[npmlx:-npmlx,0:-npmlz])/10.
    
    plot.figure(figsize = (14,4))

    extent = [fscale*x0pml,fscale*x1pml,fscale*z1pml,fscale*z0pml]
    
    fig = plot.imshow(np.transpose(U[npmlx:-npmlx,0:-npmlz]),vmin=-scale, vmax=scale, cmap="gray", aspect=1, extent=extent)
    
    plot.axis('equal')
    
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
   
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
    
    if(i==0): plot.title('Forward Displacement Profile')
    if(i==1): plot.title('Adjoint Displacement Profile')

    plot.grid()
    
    ax = plot.gca()
   
    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
    
    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
    
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    tick_locator = ticker.MaxNLocator(nbins=5)
    
    cbar = plot.colorbar(fig, cax=cax, format='%.1e')
    
    cbar.locator = tick_locator
    
    cbar.update_ticks()
    
    plot.draw()
            
    if(i==0): plot.savefig('figures/forward_displacement_map.eps',dpi=100,format='eps')
    if(i==1): plot.savefig('figures/adjoint_displacement_map.eps',dpi=100,format='eps')
    
    #plot.show()
    
    plot.close()
    
    return
#==============================================================================

#==============================================================================
# Plot 10 - Forward Difference Displacement
#==============================================================================
def graph2sdif(i,setup):
    
    x0pml = setup.x0pml
    x1pml = setup.x1pml
    z0pml = setup.z0
    z1pml = setup.z1pml
    npmlx = setup.npmlx
    npmlz = setup.npmlz
    
    if(i==0):
        
        usol = np.load("data_save/uforward_devito.npy")
        uref = np.load("data_save/uforward_reference_devito.npy")
    
    if(i==1):
        
        usol = np.load("data_save/uadjoint_devito.npy")
        uref = np.load("data_save/uadjoint_reference_devito.npy")
    
    udif = uref-usol
    
    fscale = 10**(-3) 
    
    scale  = np.amax(udif)/10.
    
    plot.figure(figsize = (14,4))

    extent = [fscale*x0pml,fscale*x1pml,fscale*z1pml,fscale*z0pml]
    
    fig = plot.imshow(np.transpose(udif),vmin=-scale, vmax=scale, cmap="gray", aspect=1, extent=extent)
    
    plot.axis('equal')
    
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
   
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
    
    if(i==0): plot.title('Difference Forward Displacement Profile')
    if(i==1): plot.title('Difference Adjoint Displacement Profile')
    
    plot.grid()
    
    ax = plot.gca()
   
    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
    
    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
    
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    tick_locator = ticker.MaxNLocator(nbins=5)
    
    cbar = plot.colorbar(fig, cax=cax, format='%.1e')
    
    cbar.locator = tick_locator
    
    cbar.update_ticks()
    
    plot.draw()
            
    if(i==0): plot.savefig('figures/forward_difference_displacement_map.eps',dpi=100,format='eps')
    if(i==1): plot.savefig('figures/adjoint_difference_displacement_map.eps',dpi=100,format='eps')

    #plot.show()
    
    plot.close()
    
    return
#==============================================================================

#==============================================================================
# Plot 2 - Receivers
#==============================================================================
def graph2drec(rec,i,setup):  
    
    x0pml = setup.x0pml
    x1pml = setup.x1pml    
    npmlx = setup.npmlx
    npmlz = setup.npmlz    
    tn = setup.tn
    t0 = setup.t0
    
    fscale = 10**(-3) 
    
    plot.figure(figsize = (14,4))
    
    #scale  = np.amax(rec[:,npmlx:-npmlx])/10.
    
    #extent = [fscale*x0pml,fscale*x1pml, fscale*tn, fscale*t0]
    
    #fig    = plot.imshow(rec[:,npmlx:-npmlx], vmin=-scale, vmax=scale, cmap="gray", extent=extent)
    
    scale  = np.amax(rec)/10.
    
    extent = [fscale*x0pml,fscale*x1pml, fscale*tn, fscale*t0]
    
    fig    = plot.imshow(rec, vmin=-scale, vmax=scale, cmap="gray", extent=extent)
    
    plot.axis('equal')
    
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
   
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f s'))
    
    if(i==0): plot.title('Receivers Profile')
    if(i==1): plot.title('Receivers Profile of Reference')
    
    plot.grid()

    ax = plot.gca()
    
    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
    
    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
    
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    tick_locator = ticker.MaxNLocator(nbins=5)
    
    cbar = plot.colorbar(fig, cax=cax, format='%.1e')
    
    cbar.locator = tick_locator
    
    cbar.update_ticks()
    
    plot.draw()
           
    if(i==0): plot.savefig('figures/receivers_map.eps',dpi=100,format='eps')
    if(i==1): plot.savefig('figures/receivers_reference_map.eps',dpi=100,format='eps')
    
    #plot.show()

    plot.close()
    
    return
#==============================================================================

#==============================================================================
# Plot 20 - Receivers Difference
#==============================================================================
def graph2drecdif(setup):  
    
    x0pml = setup.x0pml
    x1pml = setup.x1pml    
    npmlx = setup.npmlx
    npmlz = setup.npmlz    
    tn = setup.tn
    t0 = setup.t0
    
    fscale = 10**(-3) 
    
    recsol = np.load("data_save/receivers_devito.npy")
    recref = np.load("data_save/receivers_reference_devito.npy")
    recdif = recsol-recref
    
    plot.figure(figsize = (14,4))
    
    #scale  = np.amax(rec[:,npmlx:-npmlx])/10.
    
    #extent = [fscale*x0pml,fscale*x1pml, fscale*tn, fscale*t0]
    
    #fig    = plot.imshow(rec[:,npmlx:-npmlx], vmin=-scale, vmax=scale, cmap="gray", extent=extent)
    
    scale  = np.amax(recdif)/10.
    
    extent = [fscale*x0pml,fscale*x1pml, fscale*tn, fscale*t0]
    
    fig    = plot.imshow(recdif, vmin=-scale, vmax=scale, cmap="gray", extent=extent)
    
    plot.axis('equal')
    
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
   
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f s'))
    
    plot.title('Difference Receivers Profile')
    
    plot.grid()

    ax = plot.gca()
    
    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
    
    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
    
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    tick_locator = ticker.MaxNLocator(nbins=5)
    
    cbar = plot.colorbar(fig, cax=cax, format='%.1e')
    
    cbar.locator = tick_locator
    
    cbar.update_ticks()
    
    plot.draw()
           
    plot.savefig('figures/receivers_difference_map.eps',dpi=100,format='eps')
    
    #plot.show()

    plot.close()
    
    return
#==============================================================================


#==============================================================================
# Plot 3 - Gradient
#==============================================================================
def plotgrad(grad,setup):
    
    x0pml = setup.x0pml
    x1pml = setup.x1pml
    z0pml = setup.z0
    z1pml = setup.z1pml
    npmlx = setup.npmlx
    npmlz = setup.npmlz
    
    plot.figure(figsize = (14,4))
    
    fscale = 10**(-3)     
    
    extent = [fscale*x0pml,fscale*x1pml,fscale*z1pml,fscale*z0pml]
    
    vminv  = 0.9*(np.amin(grad.data)/10)
    
    vmaxv  = 1.1*(np.amax(grad.data)/10)
    
    fig=plot.imshow(np.transpose(grad[npmlx:-npmlx,0:-npmlz]),vmin=vminv,vmax=vmaxv,cmap=cm.jet,extent=extent)
    
    plot.axis('equal')
    
    ax = plot.gca()
    
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
   
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f s'))
    
    plot.title('Gradient Profile')
    
    plot.grid()

    ax = plot.gca()
    
    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
    
    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
   
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    tick_locator = ticker.MaxNLocator(nbins=5)
    
    cbar = plot.colorbar(fig, cax=cax, format='%.1e')
    
    cbar.locator = tick_locator
    
    cbar.update_ticks()
    
    plot.draw()
   
    plot.savefig('figures/grad_map.eps',dpi=100,format='eps')
    
    #plot.show()

    plot.close()
    
    return
#==============================================================================

#==============================================================================
# Plot 4 - Velocity
#==============================================================================
def graph2dvel(vel,i,setup):
   
    x0 = setup.x0
    x1 = setup.x1
    z0 = setup.z0
    z1 = setup.z1
    
    x0pml = setup.x0pml
    x1pml = setup.x1pml
    z0pml = setup.z0pml
    z1pml = setup.z1pml
    npmlx = setup.npmlx
    npmlz = setup.npmlz
    
    plot.figure(figsize = (14,4))
    
    fscale =  10**(-3)
    
    vminv = np.amin(vel)

    vmaxv = np.amax(vel)
    
    if(i==0):
        
        scale  = np.amax(vel[npmlx:-npmlx,0:-npmlz])

        extent = [fscale*x0pml,fscale*x1pml, fscale*z1pml, fscale*z0pml]
        
        fig = plot.imshow(np.transpose(vel[npmlx:-npmlx,0:-npmlz]), vmin=vminv,vmax=vmaxv, cmap=cm.jet, extent=extent)
    
    
    if(i==1):
    
        scale  = np.amax(vel)

        extent = [fscale*x0,fscale*x1,fscale*z1,fscale*z0]

        fig = plot.imshow(np.transpose(vel), vmin=vminv,vmax=vmaxv, cmap=cm.jet, extent=extent)
            
        
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))

    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))

    if(i==0): plot.title('Reduced Velocity Profile')
    if(i==1): plot.title('Full Velocity Profile')

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

    if(i==0): plot.savefig('figures/reduced_vel_map.eps',dpi=100,format='eps')
    if(i==1): plot.savefig('figures/full_vel_map.eps',dpi=100,format='eps')

    #plot.show()
    
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plot 5 - Velocity Result and True
#==============================================================================
def graph2dvel2(vel1,vel2,setup):
         
    x0pml = setup.x0pml
    x1pml = setup.x1pml
    z0pml = setup.z0pml
    z1pml = setup.z1pml
    npmlx = setup.npmlx
    npmlz = setup.npmlz
    
    plot.figure(figsize = (8,12))
    
    fscale =  10**(-3)
    
    vminv = min(np.amin(vel1),np.amin(vel2))

    vmaxv = max(np.amax(vel1),np.amax(vel2))
    
    grid = plot.GridSpec(2,1,wspace=0.4,hspace=0.4)

    extent = [fscale*x0pml,fscale*x1pml, fscale*z1pml, fscale*z0pml]
    
    plot.subplot(grid[0,0])
    
    fig = plot.imshow(np.transpose(vel1[npmlx:-npmlx,0:-npmlz]), vmin=vminv,vmax=vmaxv, cmap=cm.jet, extent=extent)
    
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
    
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
    
    plot.title('Velocity - True')
    
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
    
    plot.subplot(grid[1,0])
    
    fig = plot.imshow(np.transpose(vel2[npmlx:-npmlx,0:-npmlz]), vmin=vminv,vmax=vmaxv, cmap=cm.jet, extent=extent)
    
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
    
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
    
    plot.title('Velocity - Approximate')
    
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
        
    plot.savefig('figures/vel_map_comparative.eps',dpi=100,format='eps')

    #plot.show()
    
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plot 6 - Objective Values
#==============================================================================
def graphobjv(objv):

    plot.figure(figsize = (14,10))
    
    grid   = plot.GridSpec(2,1,wspace=0.4,hspace=0.6)

    scalex = np.amax(objv[:,0])
    
    plot.subplot(grid[0,0])
       
    scaleymin = 0.8*np.amin(objv[:,1])
    
    scaleymax = 1.2*np.amax(objv[:,1])
    
    fig = plot.plot(objv[:,0],objv[:,1],label="Objective Values")
    
    plot.legend(loc="upper right")
    
    plot.title('Objective Values per Iteration')
    
    plot.xlim((1,scalex))
    
    plot.ylim((scaleymin,scaleymax))
    
    plot.grid()
    
    plot.xlabel('Iterations')
    
    plot.ylabel('Objective Values')
    
    plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    ax = plot.gca()     
    
    ttl = ax.title
    
    ttl.set_position([.5, 1.2])

    plot.subplot(grid[1,0])
       
    scaleymin = 0.8*np.amin(objv[:,2])
    
    scaleymax = 1.2*np.amax(objv[:,2])
    
    fig = plot.plot(objv[:,0],objv[:,2],label="Relative Quadractic Norm")
    
    plot.legend(loc="upper right")
    
    plot.title('Relative Quadratic for Velocity Profile per Iteration')
    
    plot.xlim((1,scalex))
    
    plot.ylim((scaleymin,scaleymax))
    
    plot.grid()
    
    plot.xlabel('Iterations')
    
    plot.ylabel('Relative Quadractic Norm')
    
    plot.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    ax = plot.gca()

    ttl = ax.title
    
    ttl.set_position([.5, 2.2])
          
    plot.savefig('figures/obj_velnomr_values.eps',dpi=100,format='eps')

    #plot.show()
    
    plot.close()

    return
#==============================================================================

#==============================================================================
# Plot 7 - Residual Receivers
#==============================================================================
def graph2drecres(rec,setup):  
    
    x0pml = setup.x0pml
    x1pml = setup.x1pml    
    npmlx = setup.npmlx
    npmlz = setup.npmlz    
    tn = setup.tn
    t0 = setup.t0
    
    fscale = 10**(-3) 
    
    plot.figure(figsize = (14,4))
    
    scale  = np.amax(rec)/10.
    
    extent = [fscale*x0pml,fscale*x1pml, fscale*tn, fscale*t0]
    
    fig    = plot.imshow(rec, vmin=-scale, vmax=scale, cmap="seismic", extent=extent)
    
    plot.axis('equal')
    
    plot.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f km'))
   
    plot.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f s'))
    
    plot.title('Residual Receivers Profile')
    
    plot.grid()

    ax = plot.gca()
    
    ax.xaxis.set_major_locator(plot.MaxNLocator(4))
    
    ax.yaxis.set_major_locator(plot.MaxNLocator(4))
    
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    tick_locator = ticker.MaxNLocator(nbins=5)
    
    cbar = plot.colorbar(fig, cax=cax, format='%.1e')
    
    cbar.locator = tick_locator
    
    cbar.update_ticks()
    
    plot.draw()

    #plot.show()
           
    manager = plot.get_current_fig_manager()
   
    manager.full_screen_toggle()    

    plot.savefig('figures/residual_receivers_map.eps',dpi=100,format='eps')
    
    plot.close()
    
    return
#==============================================================================
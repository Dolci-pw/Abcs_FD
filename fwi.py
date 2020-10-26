#==============================================================================
# Python Imports
#==============================================================================
import numpy                                        as np
import math                                         as mt
import time                                         as tm
import sys
import segyio
import matplotlib.pyplot                            as plot
from   numpy             import linalg              as la
from   scipy.ndimage     import gaussian_filter
from   scipy             import optimize
#==============================================================================

#==============================================================================
# Devito Imports
#==============================================================================
from   devito           import *

#==============================================================================

#==============================================================================
# Our Imports
#==============================================================================
import settings_config
sys.path.insert(0, './code')
from   timeit import default_timer as timer
import solver, domain2D, utils, velmodel
from   plots  import plotgrad, graph2drec, graph2d, graph2dvel, graph2dvel2, graphobjv, graph2drecres
#==============================================================================

#==============================================================================
plot.close("all")
#==============================================================================
# Model
model=settings_config.settings.model

#==============================================================================
# Parameters Settings
#==============================================================================
if  model['vp']=='Circle':
    setting = settings_config.settings.setting6
elif model['vp']=='Marmousi':
    setting = settings_config.settings.setting5
elif model['vp']=='GM':
    setting = settings_config.settings.setting3

setup   = utils.ProblemSetup(setting)
#==============================================================================
# Grid
grid    = domain2D.SetGrid(setup)
(x, z)  = grid.dimensions
#==============================================================================
# Chosing the model
if   model['vp']=='Circle':
    vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_circle=3, vp_background=2.5,r=75)
elif model['vp']=='Marmousi':
    with segyio.open('VelModelFiles/Mar2_Vp_1.25m.segy') as segyfile:
        vp_file = segyio.tools.cube(segyfile)[0,6200:6800,100:700]
    vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)
elif model['vp']=='GM':
    vp_file = np.fromfile('VelModelFiles/gm_perfil1.bin',dtype='float32')
    vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)


#==============================================================================    
# Time Parameters
set_time = setup.TimeDiscret(v0)
#==============================================================================    
# FWI solver class
fwisolver = solver.FWISolver(set_time,setup, setting,grid,utils,v0)

#==============================================================================
# FWI Analisys Variables
#==============================================================================    
objvr = np.zeros((1000,2))
cont  = 0 
#==============================================================================    

#==============================================================================
# FWI iteration
#==============================================================================    
def shots(m0):

    global objvr, cont, v0, setup

    grad      = Function(name="grad", grid=grid)
    sd        = setting["shots_dist"]
    nshots    = int((setting["lenx"]-4*sd)/sd)
    objective = 0.
    
    if(setting["Abcs"]=='pml'):
     
        vp_guess0  = Function(name="vp_guess0",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
        vp_guess0.data[:,:] = m0.reshape((setup.nptx,setup.nptz))
   
        v1loc = utils.gerav1m0(setup,vp_guess0.data)

        vp_guess1  = Function(name="vp_guess1",grid=grid,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
        vp_guess1.data[0:setup.nptx-1,0:setup.nptz-1]  = v1loc
        vp_guess1.data[setup.nptx-1,0:setup.nptz-1]    = vp_guess1.data[setup.nptx-2,0:setup.nptz-1]
        vp_guess1.data[0:setup.nptx,setup.nptz-1]      = vp_guess1.data[0:setup.nptx,setup.nptz-2]
    
        vp_guess = [vp_guess0,vp_guess1]
    
    else:
    
        vp_guess  = Function(name="vp_guess",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
        vp_guess.data[:,:] = m0.reshape((setup.nptx,setup.nptz))
    
    for sn in range(0, nshots+1):    
    
        clear_cache()
                
        objective = objective + fwisolver.apply(sn,grad,sd,vp,vp_guess)
        
    grad_grop = np.array(grad.data[:])[:, :]
      
    if(setting["Abcs"]=='pml'):
        
        vel  = v0[0]
        vres = vp_guess[0].data 
    
    else:
    
        vel  = v0
        vres = vp_guess.data 
    
    objvr[cont,0] = objective
    
    objvr[cont,1] = la.norm(np.reshape(vres - vel,-1),2)/la.norm(np.reshape(vel,-1),2)
    
    cont = cont + 1
    
    print('The objective value in the ', cont, ' iteration is: ',np.round(objective,2))
  
    return objective, np.reshape(grad_grop,-1)
#==============================================================================    

#==============================================================================
# Initial Velocity Model

if model['vp']=='Circle':
    _, vini = velmodel.SetVel(model,setup, setting,grid,vp_circle=2.7, vp_background=2.5,r=75)
elif model['vp']=='Marmousi' or model['vp']=='GM':
    sigma  = 15 
    vini   = gaussian_filter(v0,sigma=sigma)


if(setting["Abcs"]=='pml'):
    
    m0     = np.reshape(vini[0],-1)
    vmax   = np.amax(v0[0])
    vmin   = np.amin(v0[0])
    vel    = v0[0]
    
else:
    
    m0     = np.reshape(vini,-1)
    vmax   = np.amax(v0)
    vmin   = np.amin(v0)
    vel    = v0
    
bounds = [(vmin,vmax) for _ in range(len(m0))] 


start   = tm.time()
result  = optimize.minimize(shots, m0, method='L-BFGS-B', jac=True, tol = 1e-4, bounds=bounds, options={"disp": True,"eps": 1e-4, "gtol": 1e-4,"maxiter": 5})
end     = tm.time()

vobj    = np.zeros((cont,3)) 

for i in range(0,cont):

    vobj[i,0] = i + 1
    vobj[i,1] = objvr[i,0]
    vobj[i,2] = objvr[i,1]

vresult = result.x
vresult = vresult.reshape((setup.nptx,setup.nptz))
#==============================================================================    

#==============================================================================
print("Elapsed (after compilation) = %s" % (end - start))
#==============================================================================

#==============================================================================
# Plot Results
#==============================================================================    
# P1 = plotgrad(grad.data,setup)
# P2 = graph2drec(rec.data, setup)
# P3 = graph2dvel(v0,setup)
# P4 = graph2dvel(vresul,setup)
#==============================================================================

#==============================================================================
# Plot Comprative Results
#==============================================================================    
P5 = graph2dvel2(vel,vresult,setup)
P6 = graphobjv(vobj)
#==============================================================================
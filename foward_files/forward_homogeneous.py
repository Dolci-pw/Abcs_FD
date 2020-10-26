#==============================================================================
# Python Imports
#==============================================================================
import numpy             as np
import math              as mt
import time              as tm
import sys
import matplotlib.pyplot as plot
#==============================================================================

#==============================================================================
# Devito Imports
#==============================================================================
from   devito           import *
from   examples.seismic import RickerSource
from   examples.seismic import Receiver
#==============================================================================

#==============================================================================
# Our Imports
#==============================================================================
import settings_config
sys.path.insert(0, './code')
from   timeit import default_timer as timer
import solver, domain2D, domain2Dhabc, utils, velmodel
from   plots  import plotgrad, graph2drec, graph2d, graph2dvel
#==============================================================================

#==============================================================================
plot.close("all")
#==============================================================================

#==============================================================================
# Parameters Settings for Homogeneous Model
#==============================================================================
setting = settings_config.settings.setting1
setup   = utils.ProblemSetup(setting)
#==============================================================================

#==============================================================================
# Domain and Subdomains Settings
#==============================================================================            

d0_domain = domain2D.physdomain(setup.npmlx,setup.npmlz)
d1_domain = domain2D.leftExtension(setup.npmlx,setup.npmlz)
d2_domain = domain2D.rightExtension(setup.npmlx,setup.npmlz)
d3_domain = domain2D.bottomExtension(setup.npmlx,setup.npmlz)

origin  = (setup.x0, setup.z0)
extent  = (setup.compx,setup.compz)
shape   = (setup.nptx,setup.nptz)
grid    = Grid(origin=origin,extent=extent,shape=shape,subdomains=(d0_domain,d1_domain,d2_domain,d3_domain),dtype=np.float64) 
(x, z)  = grid.dimensions
#==============================================================================

#==============================================================================
# Velocity Model
#==============================================================================
v0 = velmodel.HomogVelModel2(setup, setting["Abcs"]) 

if(setting["Abcs"]=='pml'):

    vel0 = Function(name="vel0",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
    vel0.data[:,:]  = v0[0]
    
    vel1 = Function(name="vel1", grid=grid,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
    vel1.data[0:setup.nptx-1,0:setup.nptz-1]  = v0[1]
    vel1.data[setup.nptx-1,0:setup.nptz-1]    = vel1.data[setup.nptx-2,0:setup.nptz-1]
    vel1.data[0:setup.nptx,setup.nptz-1]      = vel1.data[0:setup.nptx,setup.nptz-2]

else:

    vel0 = Function(name="vel0",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
    vel0.data[:,:] = v0
#==============================================================================

#==============================================================================
# Time Parameters
#==============================================================================    
dt0, nt, time_range = setup.TimeDiscret(v0)   #time discretization
#==============================================================================

#==============================================================================
# Source Prameters
#==============================================================================    
src = RickerSource(name='src',grid=grid,f0=setting["f0"],npoint=1,time_range=time_range,staggered=NODE,dtype=np.float64)
src.coordinates.data[:, 0] = setup.x0pml + setting["shotposition_x"] 
src.coordinates.data[:, 1] = setting["recposition_z"] 
#==============================================================================

#==============================================================================
# Receivers Parameters
#==============================================================================    
nrec = setting["rec_n"] #receivers numbers
rec  = Receiver(name='rec',grid=grid,npoint=nrec,time_range=time_range,staggered=NODE,dtype=np.float64)
rec.coordinates.data[:, 0] = np.linspace(setup.x0pml,setup.x1pml,nrec)
rec.coordinates.data[:, 1] = setting["recposition_z"] 
#==============================================================================
 
#==============================================================================
# Saves Parameters
#==============================================================================    
nsnaps = int(setting["snapshots"])
factor = mt.ceil(nt/nsnaps) + 1
time_subsampled = ConditionalDimension('t_sub',parent=grid.time_dim, factor=factor)
usave    = TimeFunction(name='usave',grid=grid,time_order=setup.tou, space_order=setup.sou,save=nsnaps,time_dim=time_subsampled,staggered=NODE,dtype=np.float64)
Usaveloc = np.zeros((nsnaps,setup.nptx,setup.nptz))
#==============================================================================

#==============================================================================
# Solver Settigns
#==============================================================================    
SolvBase = solver.solverABCs(setup,grid)

if(setting["Abcs"]=='damping'):
    solv, g, vp = settings_config.solversetting(SolvBase, setting["Abcs"], v0, setup, utils, vel0)            
elif(setting["Abcs"]=='pml'):
    solv, g, vp = settings_config.solversetting(SolvBase, setting["Abcs"], v0, setup, utils, vel0, vel1=vel1)
elif(setting["Abcs"]=='cpml'):
    solv, g, vp = settings_config.solversetting(SolvBase, setting["Abcs"], v0, setup, utils, vel0, dt0=dt0)   
elif(setting["Abcs"]=='habc-a1'):
    solv, g, vp = settings_config.solversetting(SolvBase, setting["Abcs"], v0, setup, utils, vel0, habcw=setting["habcw"])   
else:
    assert "Invalid option"
    
start    = tm.time()
rec0, u0 = solv(rec,src,vp,g,dt0,nt,system='forward',save=True,usave=usave)
end      = tm.time() 

Usaveloc[:,:,:]  = usave.data[:,:,:]
Usaveloc[-1,:,:] = u0[0,:,:]
#==============================================================================

#==============================================================================
print("Elapsed (after compilation) = %s" % (end - start))
#==============================================================================

#==============================================================================
# Plot Results
#==============================================================================    
P1 = graph2d(Usaveloc[-1,:,:],setup)
P2 = graph2drec(rec0,setup)
P3 = graph2dvel(v0[0],setup) 
#==============================================================================
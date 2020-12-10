if (__name__=='__main__'):    
    #==============================================================================
    # Python Imports
    #==============================================================================
    import numpy                                        as np
    import math                                         as mt
    import time                                         as tm
    import sys
    import segyio
    import matplotlib.pyplot                            as plot
    import psutil
    from   numpy             import linalg              as la
    from   scipy.ndimage     import gaussian_filter
    from   scipy             import optimize
    from dask.distributed    import Client, wait
    #==============================================================================

    #==============================================================================
    # Devito Imports
    #==============================================================================
    from   devito           import *
    from   examples.seismic import RickerSource
    from   examples.seismic import Receiver
    configuration['log-level']='ERROR'
    #==============================================================================

    #==============================================================================
    # Our Imports
    #==============================================================================
    import settings_config
    sys.path.insert(0, './code')
    from   timeit import default_timer as timer
    import solver, domain2D, utils, velmodel
    # from   plots  import plotgrad, graph2drec, graph2d, graph2dvel, graph2dvel2
    # from   plots  import graphobjv, graph2drecres, graph2dvelfull, graph2dadj
    # from   plots  import graph2sdif_forward, graph2sdif_adjoint
    #==============================================================================

    #==============================================================================
    plot.close("all")
    #==============================================================================
    
    #==============================================================================
    # Model Properties
    #==============================================================================    
    model=settings_config.settings.model
    #==============================================================================
    
    #==============================================================================
    # Parameters Settings
    #==============================================================================
    if(model['vp']=='Circle'):
    
        setting = settings_config.settings.setting6
    
    elif(model['vp']=='Marmousi'):
    
        setting = settings_config.settings.setting5
        
    elif(model['vp']=='Marmousi_Reference'):
    
        setting = settings_config.settings.setting51
    
    elif(model['vp']=='GM'):
    
        setting = settings_config.settings.setting3

    setup   = utils.ProblemSetup(setting)
    #==============================================================================
    
    #==============================================================================
    # Grid Construction
    #==============================================================================
    grid    = domain2D.SetGrid(setup)
    (x, z)  = grid.dimensions
    #==============================================================================
    
    #==============================================================================
    # Chosing the model
    #==============================================================================
    if(model['vp']=='Circle'):
        
        vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_circle=3, vp_background=2.5,r=75)
    
    elif(model['vp']=='Marmousi'):
    
        with segyio.open('VelModelFiles/Mar2_Vp_1.25m.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        
        vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)
        
    elif(model['vp']=='Marmousi_Reference'):
    
        with segyio.open('VelModelFiles/Mar2_Vp_1.25m.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        
        vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)
    
    elif(model['vp']=='GM'):
    
        vp_file = np.fromfile('VelModelFiles/gm_perfil1.bin',dtype='float32')
        vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)
    #==============================================================================
    # graph2dvelfull(vp.data,setup)
    # quit()
    #==============================================================================    
    # Time Parameters
    #==============================================================================
    set_time = setup.TimeDiscret(v0)
    dt0, nt, time_range = set_time   #time discretization
    #==============================================================================    

    #==============================================================================
    # Start DASK
    #==============================================================================
    if setting["dask"]: 

        from distributed import LocalCluster
        
        cluster = LocalCluster(n_workers=nshots,death_timeout=600)

        if(setting["dask"]):
            
            client = Client(cluster)
        
        else:
        
            assert "Not adapted"
    #==============================================================================

    #==============================================================================
    # FWI Solver Class
    #==============================================================================
    fwisolver = solver.FWISolver(set_time,setup, setting,grid,utils,v0,vp)
    #==============================================================================

    #==============================================================================
    # Generating true recevivers
    #==============================================================================
    rec_true  = []
    work_true = []
    
    sn = 0
    
    # Solve the forward eq. using the true model
    aux0, usave = fwisolver.forward_true(sn)
    rec_true.append(aux0)
    
    fwisolver.rec_true = rec_true
    #==============================================================================
    
    #==============================================================================
    # FWI Functions
    #==============================================================================    
    def shots(m0):

        global objvr, cont, v0, setup     
        objective = 0.

        work  = []
        grad1 = Function(name="grad1", grid=grid)
        
        clear_cache()

        #update vp_guess
        fwisolver.vp_guess(m0)
        vp_guess = fwisolver.vp_g
        
        usave, vsave  = fwisolver.apply(sn)
    
        # np.save('data_save/objvr',objvr)
        # np.save('data_save/vres',vres)
          
        return  vsave.data
    #==============================================================================    

    #==============================================================================
    # Initial Velocity Model
    #==============================================================================
    if(model['vp']=='Circle'):
        
        _, vini = velmodel.SetVel(model,setup, setting,grid,vp_circle=2.7, vp_background=2.5,r=75)
   
    elif(model['vp']=='Marmousi' or model['vp']=='GM' or model['vp']=='Marmousi_Reference'):
    
        sigma  = 15 
        vini   = gaussian_filter(v0,sigma=sigma)

    m0     = np.reshape(vini,-1)
    vmax   = np.amax(v0)
    vmin   = np.amin(v0)
    vel    = v0
   
    #==============================================================================

    #==============================================================================
    # Save Options
    #==============================================================================
    vsave = shots(m0)
    # np.save('data_save/fwd_damp',usave)
    # np.save('data_save/adj_damp',vsave)
    #usave = forward solution
    #vsave = adjoint solution
    #==============================================================================

    #==============================================================================
    # Plot Results
    #==============================================================================    
    #V1 = graph2dvel(vel,setup)
    #V2 = graph2dvelfull(vel,setup)
    #P3 = graph2d(usave[9],setup)
    #P4 = graph2dadj(vsave[1],setup)
    #==============================================================================

    #==============================================================================
    # Save Results
    #==============================================================================
    # if(model['vp']=='Marmousi_Reference'):

    #     S1 = utils.datasave(usave[9],vsave[1],setup,1)
        
    # else:
        
    #     S1 = utils.datasave(usave[9],vsave[1],setup,0)
    #==============================================================================
    
    #==============================================================================
    # Comparative Results
    #==============================================================================
    # if(model['vp']!='Marmousi_Reference'):

    #     C1 = graph2sdif_forward(setup)
    #     C2 = graph2sdif_adjoint(setup)
    #==============================================================================
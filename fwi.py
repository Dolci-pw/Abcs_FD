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
    from   dask.distributed  import Client, wait
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
    import settings_config, settings_config_refinamento 
    sys.path.insert(0, './code')
    from   timeit import default_timer as timer
    import solver, domain2D, utils, velmodel
    from   plots  import plotgrad, graph2drec, graph2d, graph2dvel, graph2dvel2
    from   plots  import graphobjv, graph2drecres
    from   plots  import graph2sdif, graph2drecdif
    #==============================================================================

    #==============================================================================
    plot.close("all")
    #==============================================================================
    
    #==============================================================================
    # Model Properties
    #==============================================================================    
    #model = settings_config.settings.model
    model = settings_config_refinamento.settings.model
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
        
    elif(model['vp']=='Marmousi1'):
    
        setting = settings_config_refinamento.settings.setting1
        
    elif(model['vp']=='Marmousi10'):
    
        setting = settings_config_refinamento.settings.setting10
    
    elif(model['vp']=='Marmousi2'):
    
        setting = settings_config_refinamento.settings.setting2
        
    elif(model['vp']=='Marmousi20'):
    
        setting = settings_config_refinamento.settings.setting20
    
    elif(model['vp']=='Marmousi3'):
    
        setting = settings_config_refinamento.settings.setting3
        
    elif(model['vp']=='Marmousi30'):
    
        setting = settings_config_refinamento.settings.setting30
    
    elif(model['vp']=='Marmousi4'):
    
        setting = settings_config_refinamento.settings.setting4
        
    elif(model['vp']=='Marmousi40'):
    
        setting = settings_config_refinamento.settings.setting40
    
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

    elif(model['vp']=='Marmousi1' or model['vp']=='Marmousi10'):
    
        with segyio.open('VelModelFiles/marmousi_perfil1.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        
        vp, v0 = velmodel.SetVel(model,setup,setting,grid,vp_file=vp_file)
  
    elif(model['vp']=='Marmousi2' or model['vp']=='Marmousi20'):
    
        with segyio.open('VelModelFiles/marmousi_perfil1.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        
        vp, v0 = velmodel.SetVel(model,setup,setting,grid,vp_file=vp_file)
  
    elif(model['vp']=='Marmousi3' or model['vp']=='Marmousi30'):
    
        with segyio.open('VelModelFiles/marmousi_perfil1.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        
        vp, v0 = velmodel.SetVel(model,setup,setting,grid,vp_file=vp_file)
  
    elif(model['vp']=='Marmousi4' or model['vp']=='Marmousi40'):
    
        with segyio.open('VelModelFiles/marmousi_perfil1.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        
        vp, v0 = velmodel.SetVel(model,setup,setting,grid,vp_file=vp_file)
    
    elif(model['vp']=='Marmousi' or model['vp']=='Marmousi_Reference'):
    
        with segyio.open('VelModelFiles/marmousi_perfil1.segy') as segyfile:
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
        
        vsave, receivers_field = fwisolver.apply(sn)
    
        # np.save('data_save/objvr',objvr)
        # np.save('data_save/vres',vres)
          
        return  vsave.data, receivers_field
    #==============================================================================    

    #==============================================================================
    # Initial Velocity Model
    #==============================================================================
    if(model['vp']=='Circle'):
        
        _, vini = velmodel.SetVel(model,setup, setting,grid,vp_circle=2.7, vp_background=2.5,r=75)
   
    elif(model['vp']=='Marmousi' or model['vp']=='GM' or model['vp']=='Marmousi_Reference'):
    
        sigma  = 15 
        vini   = gaussian_filter(v0,sigma=sigma)
    
    elif(model['vp']=='Marmousi1' or model['vp']=='Marmousi10'):
    
        sigma  = 15 
        vini   = gaussian_filter(v0,sigma=sigma)

    elif(model['vp']=='Marmousi2' or model['vp']=='Marmousi20'):
    
        sigma  = 15 
        vini   = gaussian_filter(v0,sigma=sigma)
   
    elif(model['vp']=='Marmousi3' or model['vp']=='Marmousi30'):
    
        sigma  = 15 
        vini   = gaussian_filter(v0,sigma=sigma)

    elif(model['vp']=='Marmousi4' or model['vp']=='Marmousi40'):
    
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
    vsave, receivers_field = shots(m0)
    uforward_position = setting['snapshots'] - 1 
    uadjoint_position =  1
    uforward = usave[uforward_position]
    uadjoint = vsave[uadjoint_position]
    #usave = forward solution
    #vsave = adjoint solution
    #==============================================================================

    #==============================================================================
    # Plot Results
    #==============================================================================    
    V1 = graph2dvel(vel,0,setup)
    V2 = graph2dvel(vel,1,setup)
    P3 = graph2d(uforward,0,setup)
    P4 = graph2d(uadjoint,1,setup)
    R1 = graph2drec(receivers_field,1,setup)
    #==============================================================================

    #==============================================================================
    # Save Results
    #==============================================================================
    if(model['vp']=='Marmousi' or model['vp']=='Marmousi10' or model['vp']=='Marmousi20' or model['vp']=='Marmousi30' or model['vp']=='Marmousi40'):

        S1 = utils.datasave(uforward,uadjoint,receivers_field,setup,1)
        
    else:
        
        S1 = utils.datasave(uforward,uadjoint,receivers_field,setup,0)
    #==============================================================================
    
    #==============================================================================
    # Comparative Results
    #==============================================================================
    if(model['vp']=='Marmousi' or model['vp']=='Marmousi1' or model['vp']=='Marmousi2' or model['vp']=='Marmousi3' or model['vp']=='Marmousi4'):

        C1 = graph2sdif(0,setup)
        C2 = graph2sdif(1,setup)
        C3 = graph2drecdif(setup)
    #==============================================================================
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
    from   plots  import plotgrad, graph2drec, graph2d, graph2dvel, graph2dvel2, graphobjv, graph2drecres
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

    if(model['vp']=='Marmousi'):
    
        setting = settings_config.settings.setting1

    elif(model['vp']=='GMnew'):

        setting = settings_config.settings.setting2

    elif(model['vp']=='Ovethrust'):

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
    #==============================================================================
    # Chosing the model
    #==============================================================================
    if(model['vp']=='Marmousi'):
    
        with segyio.open('VelModelFiles/Mar2_Vp_1.25m.segy') as segyfile:
            vp_file = segyio.tools.cube(segyfile)[0,:,:]
        
        vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file,start_model='True')

    elif(model['vp']=='Ovethrust'):

        vp_file = np.load("./VelModelFiles/overtrhust_2D.npy")
        vp_files = np.transpose(vp_file)
        vp, v0  = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)

    # import matplotlib.pyplot as plt     
    # c = plt.imshow(vp.data, cmap ='jet')
    # plt.colorbar(c)
    # plt.savefig("fig.png")
    # plt.show()
    # quit()
    #==============================================================================    
    # Time Parameters
    #==============================================================================
    set_time = setup.TimeDiscret(v0)
    dt0, nt, time_range = set_time   #time discretization
    #==============================================================================    
    #==============================================================================
    # Shot Properties
    #==============================================================================
    sd     = setting["shots_dist"]
    nshots = int((setting["lenx"]-200)/sd)+1
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
 
    for sn in range(0, nshots): 

        rec, u = fwisolver.forward_true(sn)
        np.save('data_save/rec5_cpml_sigma10' + str(sn),rec.data)
        np.save('data_save/u5_cpml_sigma10' + str(sn),u)
        


    


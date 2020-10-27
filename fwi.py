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
    if(model['vp']=='Circle'):
    
        setting = settings_config.settings.setting6
    
    elif(model['vp']=='Marmousi'):
    
        setting = settings_config.settings.setting5
    
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
            vp_file = segyio.tools.cube(segyfile)[0,6200:6800,100:700]
        
        vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)
    
    elif(model['vp']=='GM'):
    
        vp_file = np.fromfile('VelModelFiles/gm_perfil1.bin',dtype='float32')
        vp, v0 = velmodel.SetVel(model,setup, setting,grid,vp_file=vp_file)
    #==============================================================================
    
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
    nshots = int((setting["lenx"]-sd)/sd) 
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
    
        if(setting["dask"]):
            
            work_true.append(client.submit(fwisolver.forward_true,sn))
        
        else:
        
            rec_true.append(fwisolver.forward_true(sn))
    
    if(setting["dask"]):
        
        wait(work_true)
        
        for i in range(nshots):
        
            rec_true.append(work_true[i].result())
           
        if psutil.virtual_memory().percent > setting["memory"]:

            client.restart()

    fwisolver.rec_true = rec_true
    #==============================================================================

    #==============================================================================
    # FWI Analisys Variables
    #==============================================================================    
    objvr = np.zeros((1000,2))
    cont  = 0 
    #==============================================================================    
    
    #==============================================================================
    # FWI Functions
    #==============================================================================    
    def shots(m0):

        global objvr, cont, v0, setup     
        objective = 0.
         
        #graph2dvel(vp_guess.data[:],setup)
        
        work  = []
        grad1 = Function(name="grad1", grid=grid)
        
        for sn in range(0, nshots):    
        
            clear_cache()

            #update vp_guess
            fwisolver.vp_guess(m0)
            vp_guess = fwisolver.vp_g
            
            if(setting["dask"]):
                
                
                work.append(client.submit(fwisolver.apply,sn))
            
            else:
            
                aux0, aux1       = fwisolver.apply(sn)
                objective       += aux0
                grad1.data[:,:] += aux1

        if(setting["dask"]):
        
            wait(work)
            
            for i in range(nshots):
            
                objective       += work[i].result()[0]
                grad1.data[:,:] += work[i].result()[1]

            if(psutil.virtual_memory().percent > setting["memory"]):

                client.restart()

        grad_grop = np.array(grad1.data[:])[:, :]
        
        if(setting["Abcs"]=='pml'):
            
            vel  = v0[0]
            vres = vp_guess[0].data 
        
        else:
        
            vel  = v0
            vres = vp_guess.data 
        
        objvr[cont,0] = objective
        
        objvr[cont,1] = la.norm(np.reshape(vres - vel,-1),2)/la.norm(np.reshape(vel,-1),2)
        
        cont = cont + 1

        # np.save('data_save/objvr',objvr)
        # np.save('data_save/vres',vres)
        print('The objective value in the ', cont, ' iteration is: ',np.round(objective,2))
    
        return objective, np.reshape(grad_grop,-1)
    #==============================================================================    

    #==============================================================================
    # Initial Velocity Model
    #==============================================================================
    if(model['vp']=='Circle'):
        
        _, vini = velmodel.SetVel(model,setup, setting,grid,vp_circle=2.7, vp_background=2.5,r=75)
   
    elif(model['vp']=='Marmousi' or model['vp']=='GM'):
    
        sigma  = 20 
        vini   = gaussian_filter(v0,sigma=sigma)

    m0     = np.reshape(vini,-1)
    vmax   = np.amax(v0)
    vmin   = np.amin(v0)
    vel    = v0
    #==============================================================================

    #==============================================================================
    # FWI Interactions
    #==============================================================================
    bounds = [(vmin,vmax) for _ in range(len(m0))] 

    start   = tm.time()
    result  = optimize.minimize(shots, m0, method='L-BFGS-B', jac=True, tol = 1e-4, bounds=bounds, options={"disp": True,"eps": 1e-4, "gtol": 1e-4,"maxiter": 20})
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
    P3 = graph2dvel(v0,setup)
    # P4 = graph2dvel(vresul,setup)
    #==============================================================================

    #==============================================================================
    # Plot Comprative Results
    #==============================================================================    
    P5 = graph2dvel2(vel,vresult,setup)
    P6 = graphobjv(vobj)
    #==============================================================================
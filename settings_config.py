#==============================================================================
# Python Imports
#==============================================================================
import numpy             as np
#==============================================================================

#==============================================================================
class settings:
#==============================================================================
    model = {
        "vp":'Ovethrust'   # Marmousi, GMnew or Ovethrust  
            }

#==============================================================================
# Parameters Settings for Marmousi with FWI
#==============================================================================
    setting1 = {
       "x0": 4000,              # x initial in metters
        "z0": 0.,                # z initial in metters
        "lenpmlx": 400,          # pml lenght x direction 
        "lenpmlz": 400,          # pml lenght z direction 
        "nptx": 1101,             # number of points in x-axis
        "nptz": 351,             # number of points in z-axis
        "lenx": 11000,            # x-axis lenght (metters)
        "lenz": 3500,            # z-axis lenght (metters)
        "t0": 0.,                # initial time
        "tn": 5000.,             # final time milliseconds
        "CFL": 0.4,              # cfl parameter
        "f0": 0.005,              # frequency peak KHz
        "Abcs": 'cpml',           # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,   # shot position from the z0 (metters)
        "recposition_z": 2.25,   # Receiver position from the z0 (metters)
        "rec_n": 551,            # Receiver number
        "shot_n":20,              # Shot number
        "habcw": 2,              # 1=linear , 2=nonlinear weight (used in habc-a1)
        "jump": 5,               # Jump to save the wave equation solution to be used in adjoint-based gradient
        "shots_dist": 500,       # distance between the shots in metters
        "snapshots": 10,         # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,   # True or False
        "threads_per_worker": 1,
        "memory": 70.,           # Restart DASK cluster when more than X% of memory is used
        "dask": False,            # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,    # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        "multiscale":True,      # Frequency multiscale: True or False
        "freq_bands": [5, 8, 10], # frequence band
        "Wavelet_filter": False   # True or False
        }
#==============================================================================

#==============================================================================
# Parameters Settings
#==============================================================================
    setting2 = {
        "x0": 1000.,             # x initial in metters
        "z0": 0.,                # z initial in metters
        "lenpmlx": 1000,         # pml lenght x direction 
        "lenpmlz": 1000,         # pml lenght z direction 
        "nptx": 758,             # number of points in x-axis
        "nptz": 400,             # number of points in z-axis
        "lenx": 7462.5,          # x-axis lenght (metters)
        "lenz": 7992.0,          # z-axis lenght (metters)
        "t0": 0.,                # initial time
        "tn": 8000.,             # final time milliseconds
        "cfl": 0.4,              # cfl parameter
        "f0": 0.005,             # frequency peak KHz
        "Abcs": 'cpml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_x":125,    # shot position from the x0 (metters)
        "shotposition_z":7.5,    # shot position from the z0 (metters)
        "recposition_x": 2.25,   # Receiver position from the z0 (metters)
        "recposition_z": 12,     # Receiver position from the z0 (metters)
        "rec_n": 300,            # Receiver number
        "habcw": 2,              # 1=linear , 2=nonlinear weight (used in habc-a1)
        "jump": 1,               # Jump to save the wave equation solution to be used in adjoint-based gradient
        "shots_dist": 500,       # distance between the shots in metters
        "snapshots": 10,         # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,   # True or False
        "threads_per_worker": 1,
        "memory": 70.,           # Restart DASK cluster when more than X% of memory is used
        "dask": False,           # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        "multiscale": True,      # Frequency multiscale: True or False
        "freq_bands": [2, 5, 8], # frequence band
        "Wavelet_filter": True   # True or False
        }
#==============================================================================

    setting3 = {
        "x0": 0.,               # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx":0,          # pml lenght x direction (in metters?)
        "lenpmlz":0,          # pml lenght z direction (in metters?)
        "nptx": 801,            # number of points in x-axis
        "nptz": 401,            # number of points in z-axis
        "lenx": 20000,            # x-axis lenght (metters)
        "lenz": 4000,            # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 5000,              # final time milliseconds
        "f0" : 0.005,           # frequency peak kHz
        "Abcs": 'damping',      # Abcs methods, options=damping, pml, cpml or habc-a1
        "shotposition_z":2.5,   # shot position from the z0 (metters)
        "recposition_z": 5,   # Receiver position from the z0 (metters)
        "rec_n": 551,            # Receiver number
        "habcw": 2,              # 1=linear , 2=nonlinear weight (used in habc-a1)
        "jump": 5,               # Jump to save the wave equation solution to be used in adjoint-based gradient
        "shots_dist": 1000,       # distance between the shots in metters
        "snapshots": 10,         # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,   # True or False
        "threads_per_worker": 1,
        "memory": 70.,           # Restart DASK cluster when more than X% of memory is used
        "dask": False,            # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,    # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        "multiscale":True,      # Frequency multiscale: True or False
        "freq_bands": [5, 8, 10], # frequence band
        "Wavelet_filter": False   # True or False
    }
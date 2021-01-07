#==============================================================================
# Python Imports
#==============================================================================
import numpy             as np
#==============================================================================

#==============================================================================
class settings:
#==============================================================================

#==============================================================================
    model = {
        "vp":'Marmousi1'   # Circle, Marmousi, GM or GMnew   
            }
#==============================================================================
    
#==============================================================================
# Parameters Settings for Marmousi with FWI
#==============================================================================
    setting1 = {
        "x0": 4000.,            # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 900,         # pml lenght x direction 
        "lenpmlz": 320,         # pml lenght z direction 
        "nptx": 901,            # number of points in x-axis
        "nptz": 321,            # number of points in z-axis
        "lenx": 9000,           # x-axis lenght (metters)
        "lenz": 3200,           # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 3200.,            # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.02,             # frequency peak KHz
        "Abcs": 'pml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "pos_rec_0":4100,       # first position of the receiver
        "pos_rec_n":12900,      # last position of the receiver
        "rec_n": 541,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "position_src": 4500,   # source position on the physical domain
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask": False,          # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        }
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi with FWI - Reference for Setting1
#==============================================================================
    setting10 = {
        "x0": 1000.,            # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 3000,        # pml lenght x direction 
        "lenpmlz": 1240,        # pml lenght z direction 
        "nptx": 1501,           # number of points in x-axis
        "nptz": 621,            # number of points in z-axis
        "lenx": 15000,          # x-axis lenght (metters)
        "lenz": 6200,           # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 3200.,            # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.02,             # frequency peak KHz
        "Abcs": 'pml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "pos_rec_0":4100,       # first position of the receiver
        "pos_rec_n":12900,      # last position of the receiver
        "rec_n": 541,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "position_src": 7500,   # source position on the physical domain
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask": False,          # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        }
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi with FWI
#==============================================================================
    setting2 = {
        "x0": 4000.,            # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 900,         # pml lenght x direction 
        "lenpmlz": 320,         # pml lenght z direction 
        "nptx": 1801,            # number of points in x-axis
        "nptz": 641,            # number of points in z-axis
        "lenx": 9000,           # x-axis lenght (metters)
        "lenz": 3200,           # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 3200.,            # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.02,             # frequency peak KHz
        "Abcs": 'pml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "pos_rec_0":4100,       # first position of the receiver
        "pos_rec_n":12900,      # last position of the receiver
        "rec_n": 1081,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "position_src": 4500,   # source position on the physical domain
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask": False,          # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        }
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi with FWI - Reference for Setting2
#==============================================================================
    setting20 = {
        "x0": 1000.,            # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 3000,        # pml lenght x direction 
        "lenpmlz": 1240,        # pml lenght z direction 
        "nptx": 3001,           # number of points in x-axis
        "nptz": 1241,            # number of points in z-axis
        "lenx": 15000,          # x-axis lenght (metters)
        "lenz": 6200,           # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 3200.,            # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.02,             # frequency peak KHz
        "Abcs": 'pml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "pos_rec_0":4100,       # first position of the receiver
        "pos_rec_n":12900,      # last position of the receiver
        "rec_n": 1081,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "position_src": 7500,   # source position on the physical domain
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask": False,          # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        }
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi with FWI
#==============================================================================
    setting3 = {
        "x0": 4000.,            # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 900,         # pml lenght x direction 
        "lenpmlz": 320,         # pml lenght z direction 
        "nptx": 3601,            # number of points in x-axis
        "nptz": 1281,            # number of points in z-axis
        "lenx": 9000,           # x-axis lenght (metters)
        "lenz": 3200,           # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 3200.,            # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.02,             # frequency peak KHz
        "Abcs": 'pml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "pos_rec_0":4100,       # first position of the receiver
        "pos_rec_n":12900,      # last position of the receiver
        "rec_n": 2161,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "position_src": 4500,   # source position on the physical domain
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask": False,          # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        }
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi with FWI - Reference for Setting3
#==============================================================================
    setting30 = {
        "x0": 1000.,            # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 3000,        # pml lenght x direction 
        "lenpmlz": 1240,        # pml lenght z direction 
        "nptx": 6001,           # number of points in x-axis
        "nptz": 2481,            # number of points in z-axis
        "lenx": 15000,          # x-axis lenght (metters)
        "lenz": 6200,           # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 3200.,            # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.02,             # frequency peak KHz
        "Abcs": 'pml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "pos_rec_0":4100,       # first position of the receiver
        "pos_rec_n":12900,      # last position of the receiver
        "rec_n": 2161,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "position_src": 7500,   # source position on the physical domain
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask": False,          # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        }
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi with FWI
#==============================================================================
    setting4 = {
        "x0": 4000.,            # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 900,         # pml lenght x direction 
        "lenpmlz": 320,         # pml lenght z direction 
        "nptx": 7201,            # number of points in x-axis
        "nptz": 2561,            # number of points in z-axis
        "lenx": 9000,           # x-axis lenght (metters)
        "lenz": 3200,           # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 3200.,            # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.02,             # frequency peak KHz
        "Abcs": 'pml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "pos_rec_0":4100,       # first position of the receiver
        "pos_rec_n":12900,      # last position of the receiver
        "rec_n": 4231,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "position_src": 4500,   # source position on the physical domain
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask": False,          # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        }
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi with FWI - Reference for Setting4
#==============================================================================
    setting40 = {
        "x0": 1000.,            # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 3000,        # pml lenght x direction 
        "lenpmlz": 1240,        # pml lenght z direction 
        "nptx": 12001,           # number of points in x-axis
        "nptz": 4961,            # number of points in z-axis
        "lenx": 15000,          # x-axis lenght (metters)
        "lenz": 6200,           # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 3200.,            # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.02,             # frequency peak KHz
        "Abcs": 'pml',          # Abcs methods, options=damping, pml, cpml, habc-a1, Higdon
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "pos_rec_0":4100,       # first position of the receiver
        "pos_rec_n":12900,      # last position of the receiver
        "rec_n": 4231,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "position_src": 7500,   # source position on the physical domain
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask": False,          # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":False,   # True or False
        "n_checkpointing": 400,  # None or an int value n<timestep
        }
#==============================================================================
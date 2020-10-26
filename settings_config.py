#==============================================================================
# Python Imports
#==============================================================================
import numpy             as np
#==============================================================================

#==============================================================================
class settings:
#==============================================================================
    model = {
        "vp":'Marmousi'   # Circle, Marmousi or GM     
            }
#==============================================================================
# Parameters Settings for Homogeneous and Heterogeneous Model
#==============================================================================
    setting1 = {
        "x0": 0.,              # x initial in metters
        "z0": 0.,              # z initial in metters
        "lenpmlx": 200,        # pml lenght x direction  
        "lenpmlz": 200,        # pml lenght z direction 
        "nptx": 201,           # number of points in x-axis
        "nptz": 201,           # number of points in z-axis
        "lenx": 1000,          # x-axis lenght (metters)
        "lenz": 1000,          # z-axis lenght (metters)
        "t0": 0.,              # initial time
        "tn": 1000,            # final time milliseconds
        "cfl": 0.4,            # cfl parameter
        "f0" : 0.01,           # frequency peak kHz
        "Abcs": 'damping',         # Abcs methods, options=damping, pml, or habc-a1
        "shotposition_x":500,  # shot position from the x0 (metters)
        "shotposition_z":10,   # shot position from the z0 (metters)
        "recposition_x": 20,   # Receiver position from the x0 (metters)
        "recposition_z": 20,   # Receiver position from the z0 (metters)
        "rec_n": 100,          # Receiver number
        "habcw": 2,            # 1=linear , 2=nonlinear weight (used in habc-a1)
        "snapshots": 10,       # wave equation solution snapshots to be saved
        }
#==============================================================================

#==============================================================================
# Parameters Settings for GM Model
#==============================================================================
    setting2 = {
        "x0": 26650.,          # x initial in metters
        "z0": 0.,              # z initial in metters
        "lenpmlx": 1500,       # pml lenght x direction  
        "lenpmlz": 700,        # pml lenght z direction
        "nptx": 376,           # number of points in x-axis
        "nptz": 500,           # number of points in z-axis
        "lenx": 15000,         # x-axis lenght (metters)
        "lenz": 7000,          # z-axis lenght (metters)
        "t0": 0.,              # initial time
        "tn": 7000.,           # final time milliseconds
        "cfl": 0.4,            # cfl parameter
        "f0": 0.01,            # frequency peak KHz
        "Abcs": 'pml',         # Abcs methods, options=damping, pml, or habc-a1
        "shotposition_x":7500, # shot position from the x0 (metters)
        "shotposition_z":32,   # shot position from the z0 (metters)
        "recposition_x": 20,   # Receiver position from the x0 (metters)
        "recposition_z": 32,   # Receiver position from the z0 (metters)
        "rec_n": 376,          # Receiver number
        "habcw": 2,            # 1=linear , 2=nonlinear weight (used in habc-a1)
        "snapshots": 10,       # wave equation solution snapshots to be saved
        }
#==============================================================================

#==============================================================================
# Parameters Settings for GM Model with FWI
#==============================================================================
    setting3 = {
        "x0": 26650.,          # x initial in metters
        "z0": 0.,              # z initial in metters
        "lenpmlx": 1000,       # pml lenght x direction  
        "lenpmlz": 1000,       # pml lenght z direction
        "nptx": 400,           # number of points in x-axis
        "nptz": 700,           # number of points in z-axis
        "lenx": 7000,          # x-axis lenght (metters)
        "lenz": 7000,          # z-axis lenght (metters)
        "t0": 0.,              # initial time
        "tn": 7000.,           # final time milliseconds
        "cfl": 0.4,            # cfl parameter
        "f0": 0.01,           # frequency peak KHz
        "Abcs": 'damping',     # Abcs methods, options=damping, pml, or habc-a1
        "shotposition_x":32,   # shot position from the x0 (metters)
        "shotposition_z":32,   # shot position from the z0 (metters)
        "recposition_x": 32,   # Receiver position from the x0 (metters)
        "recposition_z": 32,   # Receiver position from the z0 (metters)
        "rec_n": 300,          # Receiver number
        "habcw": 2,            # 1=linear , 2=nonlinear weight (used in habc-a1)
        "jump": 5,             # Jump to save the wave equation solution to be used in adjoint-based gradient
        "shots_dist":1000,      # distance between the shots in metters
        "checkpointing":True   # True or False
        }
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi
#==============================================================================
    setting4 = {
        "x0": 0.,              # x initial in metters
        "z0": 0.,              # z initial in metters
        "lenpmlx": 25,         # pml lenght x direction  
        "lenpmlz": 25,         # pml lenght z direction 
        "nptx": 200,           # number of points in x-axis
        "nptz": 200,           # number of points in z-axis
        "lenx": 250,           # x-axis lenght (metters)
        "lenz": 250,           # z-axis lenght (metters)
        "t0": 0.,              # initial time
        "tn": 350.,            # final time milliseconds
        "cfl": 0.4,            # cfl parameter
        "f0": 0.01,            # frequency peak KHz
        "Abcs": 'pml',         # Abcs methods, options=damping, pml, cpml or habc-a1
        "shotposition_x":125,  # shot position from the x0 (metters)
        "shotposition_z":125,  # shot position from the z0 (metters)
        "recposition_x": 32,   # Receiver position from the x0 (metters)
        "recposition_z": 2.5,  # Receiver position from the z0 (metters)
        "rec_n": 200,          # Receiver number
        "habcw": 2,            # 1=linear , 2=nonlinear weight (used in habc-a1)
        "jump": 2,             # Jump to save the wave equation solution to be used in adjoint-based gradient
        "shots_dist": 1.25,    # distance between the shots in metters
        "snapshots": 10,       # wave equation solution snapshots to be saved
        }   
#==============================================================================

#==============================================================================
# Parameters Settings for Marmousi with FWI
#==============================================================================
    setting5 = {
        "x0": 0.,               # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 40,          # pml lenght x direction 
        "lenpmlz": 40,          # pml lenght z direction 
        "nptx": 200,            # number of points in x-axis
        "nptz": 200,            # number of points in z-axis
        "lenx": 200,            # x-axis lenght (metters)
        "lenz": 200,            # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 400.,             # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0": 0.005,            # frequency peak KHz
        "Abcs": 'damping',         # Abcs methods, options=damping, pml, cpml or habc-a1
        "shotposition_x":125,   # shot position from the x0 (metters)
        "shotposition_z":1.25,  # shot position from the z0 (metters)
        "recposition_x": 2.25,  # Receiver position from the z0 (metters)
        "recposition_z": 2.25,  # Receiver position from the z0 (metters)
        "rec_n": 200,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "jump": 1,              # Jump to save the wave equation solution to be used in adjoint-based gradient
        "shots_dist": 40,       # distance between the shots in metters
        "snapshots": 10,        # wave equation solution snapshots to be saved  
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask_client": False,   # This variable change if you start the DASK cluster
        "death_timeout": 60,
        "checkpointing":True    # True or False
        }
#==============================================================================

#==============================================================================
# Parameters Settings for FWI with Circle
#==============================================================================
    setting6 = {
        "x0": 0.,               # x initial in metters
        "z0": 0.,               # z initial in metters
        "lenpmlx": 100,         # pml lenght x direction
        "lenpmlz": 100,         # pml lenght z direction
        "nptx": 250,            # number of points in x-axis
        "nptz": 250,            # number of points in z-axis
        "lenx": 500,            # x-axis lenght (metters)
        "lenz": 500,            # z-axis lenght (metters)
        "t0": 0.,               # initial time
        "tn": 500,              # final time milliseconds
        "cfl": 0.4,             # cfl parameter
        "f0" : 0.005,           # frequency peak kHz
        "Abcs": 'damping',      # Abcs methods, options=damping, pml, cpml or habc-a1
        "shotposition_x":500,   # shot position from the x0 (metters)
        "shotposition_z":30,    # shot position from the z0 (metters)
        "recposition_x": 20,    # Receiver position from the x0 (metters)
        "recposition_z": 20,    # Receiver position from the z0 (metters)
        "rec_n": 200,           # Receiver number
        "habcw": 2,             # 1=linear , 2=nonlinear weight (used in habc-a1)
        "snapshots": 10,        # wave equation solution snapshots to be saved
        "jump": 1,              # Jump to save the wave equation solution to be used in adjoint-based gradient
        "shots_dist":100,        # distance between the shots in metters
        "USE_GPU_DASK": False,  # True or False
        "threads_per_worker": 1,
        "memory": 70.,          # Restart DASK cluster when more than X% of memory is used
        "dask_client": True,    # This variable change if you start the DASK cluster
        "death_timeout": 800,    
        "checkpointing":True    # True or False
        }
#==============================================================================

 
    

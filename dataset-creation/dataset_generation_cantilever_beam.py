from topopt_cholmod_cantilever_beam import topopt
import numpy as np


nelx = 180
nely = 60
rmin = 5.4
penal = 3.0
ft = 0  # ft==0 -> sens, ft==1 -> dens

num_volfrac = 2
num_load_position = 2

volfrac = np.linspace(0.4, 0.5, num=num_volfrac)  # 7 evenly spaced values between 0.2 and 0.8
load_position = np.linspace(0, 1, num=num_load_position)  # 11 evenly spaced values between 0 and 1

for i, vf in enumerate(volfrac):
    for j, lp in enumerate(load_position):
        load_config = {
            'position': lp.item(),
            'direction': 'vertical',
            'magnitude': -1.0
        }
        # Call the topology optimization function
        xPhys, obj = topopt(nelx, nely, vf.item(), penal, rmin, ft, load_config)

        # Print progress
        print(f"Processed volfrac={vf.item()}, load_position={lp.item()}, Objective={obj}")

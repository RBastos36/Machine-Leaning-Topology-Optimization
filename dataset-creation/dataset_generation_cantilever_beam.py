from topopt_cholmod_cantilever_beam import topopt
import numpy as np
import h5py

nelx = 180
nely = 60
rmin = 5.4
penal = 3.0
ft = 0  # ft==0 -> sens, ft==1 -> dens

num_volfrac = 2
num_load_position = 2

volfrac = np.linspace(0.4, 0.5, num=num_volfrac)  # 7 evenly spaced values between 0.2 and 0.8
load_position = np.linspace(0, 1, num=num_load_position)  # 11 evenly spaced values between 0 and 1

# Create an HDF5 file to save data
with h5py.File('topopt_results.h5', 'w') as hf:
    # Create datasets for results
    xPhys_dataset = hf.create_dataset('optimShape', shape=(num_volfrac, num_load_position, nely, nelx), dtype='float32')
    obj_dataset = hf.create_dataset('obj', shape=(num_volfrac, num_load_position), dtype='float32')

    # Nested loop for volfrac and load_position
    for i, vf in enumerate(volfrac):
        for j, lp in enumerate(load_position):
            load_config = {
                'position': lp.item(),
                'direction': 'vertical',
                'magnitude': -1.0
            }
            # Call the topology optimization function
            xPhys, obj = topopt(nelx, nely, vf.item(), penal, rmin, ft, load_config)

            # Reshape or process xPhys if needed
            xPhys_reshaped = xPhys.reshape((nelx, nely)).T

            # TODO Fix xPhys saving to dataset / Check if the xPhys matrices are correct
            # Save results into the HDF5 datasets
            xPhys_dataset[i, j, :, :] = xPhys_reshaped
            obj_dataset[i, j] = obj

            # Print progress
            print(f"Processed volfrac={vf.item()}, load_position={lp.item()}, Objective={obj}")

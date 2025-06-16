# Author: Ricardo A. O. Bastos
# Created: June 2025


import h5py
import matplotlib.pyplot as plt

plot_all = 1 # 0 to plot only the first result, 1 to plot all results

if plot_all == 0:
    # Open the HDF5 file
    with h5py.File('topopt_results.h5', 'r') as hf:
        # Extract first result
        first_result = hf['optimShape'][1, 1]
        print(first_result)

        # Create figure
        plt.figure(figsize=(16, 8))
        plt.imshow(first_result, cmap='binary')
        plt.title('First Topology Optimization Result')
        plt.axis('off')
        plt.show()

elif plot_all == 1:
    # Open the HDF5 file
    with h5py.File('topopt_results.h5', 'r') as hf:
        # Get the xPhys dataset
        xPhys = hf['optimShape'][:]

        # Get dimensions
        num_volfrac, num_load_position, nely, nelx = xPhys.shape

        # Create a figure with subplots
        fig, axes = plt.subplots(num_volfrac, num_load_position, figsize=(16, 8))

        # Flatten axes if needed
        if num_volfrac > 1 and num_load_position > 1:
            axes = axes.flatten()

        # Iterate through all results
        for i in range(num_volfrac):
            for j in range(num_load_position):
                # Select subplot
                ax = axes[i * num_load_position + j] if num_volfrac > 1 or num_load_position > 1 else axes

                # Plot the result
                ax.imshow(xPhys[i, j, :, :], cmap='binary')
                ax.set_title(f'Volfrac={i}, Load Pos={j}')
                ax.axis('off')

        plt.tight_layout()
        plt.show()
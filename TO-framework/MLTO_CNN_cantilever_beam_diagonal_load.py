from __future__ import division
import math
import os

import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import h5py
import torch
import json
from torch.utils.data import Dataset

from CNN_model_simple_node_level import TopologyOptimizationCNN


class FEMDataset(Dataset):
    def __init__(self, stats=None, data=None, device=None):
        self.stats = stats
        self.data = data
        self.device = device

    def normalize_tensor(self, tensor, is_input=True):
        """Normalize tensor using pre-computed statistics"""
        if self.stats is None:
            return tensor

        # Ensure tensor is on the same device as means and stds
        tensor = tensor.to(self.device)

        stats_key = 'inputs' if is_input else 'outputs'
        means = torch.tensor(self.stats[stats_key]['means'], device=self.device).view(-1, 1, 1)
        stds = torch.tensor(self.stats[stats_key]['stds'], device=self.device).view(-1, 1, 1)

        return (tensor - means) / (stds + 1e-8)  # Add epsilon for numerical stability

    def denormalize_tensor(self, tensor, is_input=True):
        """Denormalize tensor back to original scale"""
        if self.stats is None:
            return tensor

        stats_key = 'inputs' if is_input else 'outputs'
        means = torch.tensor(self.stats[stats_key]['means'], device=self.device).view(-1, 1, 1)
        stds = torch.tensor(self.stats[stats_key]['stds'], device=self.device).view(-1, 1, 1)

        return (tensor * stds) + means


def load_pretrained_model(model_path, device=None):
    print(f"\n--- Attempting to load model from {model_path} ---")

    # Ensure absolute path
    model_path = os.path.abspath(model_path)

    # Check if file exists with multiple checks
    if not os.path.exists(model_path):
        print(f"ERROR: Model file does not exist at {model_path}")
        print("Current working directory:", os.getcwd())
        print("Absolute path checked:", model_path)
        print("Files in current directory:", os.listdir())
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Use GPU if available, otherwise CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Print file details
        file_stats = os.stat(model_path)
        print(f"Model file size: {file_stats.st_size} bytes")
        print(f"Last modified: {os.path.getmtime(model_path)}")

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Print checkpoint keys
        print("Checkpoint keys:", list(checkpoint.keys()))

        # Verify model state dict exists
        if 'model_state_dict' not in checkpoint:
            print("ERROR: No 'model_state_dict' in checkpoint")
            print("Available keys:", list(checkpoint.keys()))
            raise ValueError("Invalid checkpoint format")

        # Initialize the model architecture
        from CNN_model_simple_node_level import TopologyOptimizationCNN
        model = TopologyOptimizationCNN()

        # Detailed state dict loading
        print("\nLoading state dict...")
        try:
            # Try loading with some flexibility
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as load_error:
            print(f"ERROR loading state dict: {load_error}")

            # Print out keys for debugging
            print("\nCheckpoint state dict keys:")
            for k in checkpoint['model_state_dict'].keys():
                print(k)

            print("\nModel state dict keys:")
            for k in model.state_dict().keys():
                print(k)

            raise

        # Set the model to evaluation mode
        model.eval()

        print("Model successfully loaded!")
        return model.to(device)

    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def calculate_dataset_statistics(inputs, device=None):
    """
    Calculate channel-wise mean and std statistics for inputs.

    Args:
        inputs (np.ndarray): Input data with shape (N, 5, height, width)
        device (torch.device, optional): Device to use for calculations

    Returns:
        dict: Dictionary containing statistics for inputs
    """
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs_tensor = torch.tensor(inputs, device=device)

    # Calculate means and standard deviations
    input_means = inputs_tensor.mean(dim=(0, 2, 3))
    input_stds = inputs_tensor.std(dim=(0, 2, 3))

    # Create statistics dictionary
    stats = {
        'inputs': {
            'means': input_means.tolist(),
            'stds': input_stds.tolist(),
            'channel_names': ['domain', 'fixed_x', 'fixed_y', 'loads_x', 'loads_y']
        },
        'outputs': {
            'means': [0, 0],  # Placeholder for output normalization
            'stds': [1, 1],   # Placeholder for output normalization
            'channel_names': ['displacement_x', 'displacement_y']
        }
    }

    return stats


def prepare_input_tensor(domain, fixed_x, fixed_y, loads_x, loads_y, stats, device):
    # Pad domain matrix
    padded_domain = np.zeros((domain.shape[0] + 1, domain.shape[1] + 1))
    padded_domain[:-1, :-1] = domain

    # Stack inputs into a single 5-channel tensor
    input_data = np.stack([padded_domain, fixed_x, fixed_y, loads_x, loads_y])

    # Convert to PyTorch tensor
    input_tensor = torch.FloatTensor(input_data)

    # Create dataset for normalization
    dataset = FEMDataset(stats=stats, device=device)
    input_tensor = dataset.normalize_tensor(input_tensor, is_input=True)

    # Move to device and add batch dimension
    return input_tensor.unsqueeze(0).to(device)


def predict_displacements(model, input_tensor, stats, device):
    # Predict displacements

    if model is None:
        raise ValueError("Model could not be loaded. Check the model path and architecture.")

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Create dataset for denormalization
    dataset = FEMDataset(stats=stats, device=device)
    output_tensor = dataset.denormalize_tensor(output_tensor.squeeze(0), is_input=False)

    return output_tensor.cpu().numpy()


def topopt(nelx, nely, volfrac, penal, rmin, ft, load_config, model_path=None, stats_path=None, device=None):
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])
    print(f"Load config: {load_config}")

    # Initialize dataset if needed
    initialize_dataset('cantilever-diagonal_framework.h5')

    # Open the HDF5 file at the start and keep it open
    h5file = h5py.File('cantilever-diagonal_framework.h5', 'a')

    # Load CNN model and statistics if paths are provided
    model = None
    stats = None

    if model_path and os.path.exists(model_path):
        try:
            print("Attempting to load model from path...")
            model = load_pretrained_model(model_path, device)
            print("Model successfully loaded!")
        except Exception as e:
            print(f"FAILED to load model: {e}")
            print("Continuing without CNN model...")
    else:
        print(f"Model path invalid: {model_path}")
        print("Current directory contents:", os.listdir())

    # If no stats provided, generate basic stats from initial domain
    if stats is None:
        input_data = np.stack([
            np.ones((nely, nelx)),  # domain
            np.zeros((nely, nelx)),  # fixed_x
            np.zeros((nely, nelx)),  # fixed_y
            np.zeros((nely, nelx)),  # loads_x
            np.zeros((nely, nelx))   # loads_y
        ])
        stats = calculate_dataset_statistics(input_data[np.newaxis], device)

    if model is None:
        print("WARNING: No CNN model loaded. Using fallback method or continuing without model.")

    try:
        problem_id = generate_problem_id(nelx, nely, volfrac, rmin, load_config)

        if problem_id in h5file['problems']:
            prob_group = h5file['problems'][problem_id]
        else:
            prob_group = h5file['problems'].create_group(problem_id)
            # Store parameters
            params = prob_group.create_group('parameters')
            params.attrs['nelx'] = nelx
            params.attrs['nely'] = nely
            params.attrs['volfrac'] = volfrac
            params.attrs['rmin'] = rmin
            params.attrs['load_position'] = load_config['position']
            params.attrs['horizontal_magnitude'] = load_config['horizontal_magnitude']
            params.attrs['vertical_magnitude'] = load_config['vertical_magnitude']

        # Max and min stiffness
        Emin = 1e-9
        Emax = 1.0

        # dofs:
        ndof = 2 * (nelx + 1) * (nely + 1)

        # Allocate design variables (as array), initialize and allocate sens.
        x_0 = np.ones(nely * nelx, dtype=float)
        x = volfrac * x_0
        xold = x.copy()
        xPhys = x.copy()

        g = 0  # must be initialized to use the NGuyen/Paulino OC approach
        dc = np.zeros((nely, nelx), dtype=float)

        # FE: Build the index vectors for the for coo matrix format.
        KE = lk()
        edofMat = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                edofMat[el, :] = np.array(
                    [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])

        # Construct the index pointers for the coo format
        iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 8))).flatten()

        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1

        # Finalize assembly and convert to csc format
        H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
        Hs = H.sum(1)

        # BC's and support (left edge fixed)
        dofs = np.arange(2 * (nelx + 1) * (nely + 1))
        fixed = dofs[0:2 * (nely + 1):1]  # Fix all DOFs on left edge
        free = np.setdiff1d(dofs, fixed)

        # Set up load vector
        f = np.zeros((ndof, 1))

        # Calculate load position
        rel_position = load_config['position']  # Between 0 and 1
        node_y_pos = int(rel_position * nely)
        node_index = nelx * (nely + 1) + node_y_pos

        # Apply both horizontal and vertical loads
        f[2 * node_index, 0] = load_config['horizontal_magnitude']  # Horizontal load
        f[2 * node_index + 1, 0] = load_config['vertical_magnitude']  # Vertical load

        # Solution and RHS vectors
        u = np.zeros((ndof, 1))

        # Initialize plot
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray', interpolation='none',
                       norm=colors.Normalize(vmin=-1, vmax=0))
        plt.show(block=False)

        loop = 0
        change = 1
        dv = np.ones(nely * nelx)
        dc = np.ones(nely * nelx)
        ce = np.ones(nely * nelx)

        # Split matrices into x and y components
        fixed_x, fixed_y = create_xy_matrices(fixed, nelx, nely)
        free_x, free_y = create_xy_matrices(free, nelx, nely)
        loads_x, loads_y = create_load_matrices(f, nelx, nely)

        # Save setup data
        if 'setup' not in prob_group:
            setup = prob_group.create_group('setup')

            # Save constraints
            constraints = setup.create_group('constraints')
            constraints.create_dataset('fixed_x', data=fixed_x)
            constraints.create_dataset('fixed_y', data=fixed_y)
            constraints.create_dataset('free_x', data=free_x)
            constraints.create_dataset('free_y', data=free_y)

            # Save loads
            loads = setup.create_group('loads')
            loads.create_dataset('x', data=loads_x)
            loads.create_dataset('y', data=loads_y)

            # Save initial domain
            setup.create_dataset('initial_domain', data=x_0.reshape((nelx, nely)).T)

        while change > 0.01 and loop < 2000:
            loop = loop + 1

            # CNN-based displacement prediction (if model is available)
            # Prepare input tensor
            input_tensor = prepare_input_tensor(
                x_0.reshape((nelx, nely)).T,  # domain
                fixed_x, fixed_y,  # constraints
                loads_x, loads_y,  # loads
                stats,
                device=device
            )

            # Predict displacements using CNN
            predicted_displacements = predict_displacements(model, input_tensor, stats, device)

            # Use predicted displacements
            u_x, u_y = predicted_displacements[0], predicted_displacements[1]
            print(u_x)
            u = np.zeros((ndof, 1))

            # Map back predicted displacements to DOF vector
            for x_coord in range(nelx + 1):
                for y_coord in range(nely + 1):
                    node_index = x_coord * (nely + 1) + y_coord
                    u[2 * node_index] = u_x[x_coord, y_coord]
                    u[2 * node_index + 1] = u_y[x_coord, y_coord]

            # Objective and sensitivity
            ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)).sum(1)
            obj = ((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()
            dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce

            dv[:] = np.ones(nely * nelx)
            # Sensitivity filtering:
            if ft == 0:
                dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)
            elif ft == 1:
                dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
                dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

            # Optimality criteria
            xold[:] = x
            (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)

            # Filter design variables
            if ft == 0:
                xPhys[:] = x
            elif ft == 1:
                xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]

            # Compute the change by the inf. norm
            change = np.linalg.norm(x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1), np.inf)

            # Plot to screen
            im.set_array(-xPhys.reshape((nelx, nely)).T)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
                loop, obj, (g + volfrac * nelx * nely) / (nelx * nely), change))

            # Save iteration data periodically
            if loop <= 10 or 2 ** (int(math.log2(loop))) == loop:
                iter_group_name = f'iter_{loop}'
                if iter_group_name not in prob_group:
                    iter_data = prob_group.create_group(iter_group_name)
                    iter_data.create_dataset('domain', data=xPhys.reshape((nelx, nely)).T, compression='gzip')

                    # Save separated displacement components
                    displacements = iter_data.create_group('displacements')
                    displacements.create_dataset('x', data=u_x, compression='gzip')
                    displacements.create_dataset('y', data=u_y, compression='gzip')

                    iter_data.attrs['compliance'] = float(obj)

        plt.show()
        input("Press any key...")

        # Save final results
        if 'results' not in prob_group:
            results = prob_group.create_group('results')
            results.create_dataset('final_domain', data=xPhys.reshape((nelx, nely)).T, compression='gzip')
            results.attrs['final_compliance'] = float(obj)
            results.attrs['total_iterations'] = loop

    finally:
        h5file.close()

    return xPhys, obj


def lk():
    E = 1
    nu = 0.3
    k = np.array(
        [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
         nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE


def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    xnew = np.zeros(nelx * nely)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = np.maximum(0.0,
                             np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        gt = g + np.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)


def initialize_dataset(filename='cantilever_dataset.h5'):
    """Initialize or open the main dataset file"""
    with h5py.File(filename, 'a') as hf:
        if 'problems' not in hf:
            hf.create_group('problems')


def generate_problem_id(nelx, nely, volfrac, rmin, load_config):
    """Generate a descriptive problem ID based on parameters"""
    return (f"vf{volfrac:.2f}_"
            f"pos{load_config['position']:.2f}_"
            f"dir{load_config['horizontal_magnitude']}_"
            f"mag{abs(load_config['vertical_magnitude'])}_"
            f"nelx{nelx}_"
            f"nely{nely}_"
            f"rmin{rmin:.1f}")


def create_xy_matrices(indices, nelx, nely):
    # Create empty matrices for x and y constraints
    x_matrix = np.zeros(((nelx + 1), (nely + 1)))
    y_matrix = np.zeros(((nelx + 1), (nely + 1)))

    # Convert DOF indices to node indices and identify direction
    for dof in indices:
        # Even indices are x constraints, odd indices are y constraints
        is_y_constraint = dof % 2
        # Convert DOF index to node index
        node_index = dof // 2

        # Get x, y coordinates from node index
        x_coord = node_index // (nely + 1)
        y_coord = node_index % (nely + 1)

        # Set 1s at constrained positions in appropriate matrix
        if is_y_constraint:
            y_matrix[x_coord, y_coord] = 1
        else:
            x_matrix[x_coord, y_coord] = 1

    return x_matrix.T, y_matrix.T


def create_load_matrices(force_vector, nelx, nely):
    # Create empty matrices for x and y loads
    x_loads = np.zeros(((nelx + 1), (nely + 1)))
    y_loads = np.zeros(((nelx + 1), (nely + 1)))

    # Get non-zero forces
    nonzero_dofs = np.nonzero(force_vector)[0]

    # Process each non-zero force
    for dof in nonzero_dofs:
        # Even indices are x forces, odd indices are y forces
        is_y_force = dof % 2
        # Convert DOF index to node index
        node_index = dof // 2

        # Get x, y coordinates from node index
        x_coord = node_index // (nely + 1)
        y_coord = node_index % (nely + 1)

        # Set force value at appropriate position
        # Using item() to get scalar value from array
        if is_y_force:
            y_loads[x_coord, y_coord] = force_vector[dof].item()
        else:
            x_loads[x_coord, y_coord] = force_vector[dof].item()

    return x_loads.T, y_loads.T


if __name__ == "__main__":
    # Default input parameters
    nelx = 180
    nely = 60
    volfrac = 0.5
    rmin = 5.4
    penal = 3.0
    ft = 0  # ft==0 -> sens, ft==1 -> dens

    # Example: 30-degree diagonal load pointing down and left
    magnitude = -1.0
    angle = 90  # degrees
    load_config = {
        'position': 0.5,
        'horizontal_magnitude': 20,
        'vertical_magnitude': 10
    }

    # Optional: Specify paths to CNN model and statistics
    model_path = 'topology_cnn_model.pkl'
    stats_path = 'dataset_statistics.json'  # Optional: path to JSON with normalization stats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xPhys, obj = topopt(
        nelx, nely, volfrac, penal, rmin, ft,
        load_config,
        model_path=model_path,  # Optional: can be None
        stats_path=None,  # Optional: can be None
        device=device
    )
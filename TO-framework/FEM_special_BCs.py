# A 200 line Topology Optimization code by Niels Aaage and Villads Egede Johansen, January 2013
# Updated by Niels Aage, February 2016
# Adapted by Ricardo Bastos, January 2025

from __future__ import division
import numpy as np
from matplotlib.pyplot import tight_layout
from numpy.ma.extras import union1d
from scipy.sparse import coo_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.cholmod
import h5py

# Import U-Net model
from ML_framework import *


def fea(nelx, nely, volfrac, load_config, penal, model_path, stats_path):
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print(f"Load config: {load_config}")

    # Initialize dataset if needed
    initialize_dataset('fem_special_bcs.h5')

    # Open the HDF5 file at the start and keep it open
    h5file = h5py.File('fem_special_bcs.h5', 'a')

    # ML initializations
    device = None
    with open(stats_path, 'r') as f:
        stats = json.load(f)


    # Initialize model
    model = TopologyOptimizationCNN()

    # Initialize tester
    tester = SingleInstanceTester(
        model=model,
        model_path=model_path,
        stats=stats,
        device=device
    )

    # Load trained model
    metrics = tester.load_model()

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
        E = 1
        nu = 0.3

        # dofs:
        ndof = 2 * (nelx + 1) * (nely + 1)

        # Allocate design variables (as array), initialize and allocate sens.
        x_0 = np.ones(nely * nelx, dtype=float)
        x = volfrac * x_0
        xPhys = x.copy()

        g = 0  # must be initialized to use the NGuyen/Paulino OC approach
        dc = np.ones(nely * nelx)
        dc_ml = np.ones(nely * nelx)

        # FE: Build the index vectors for the for coo matrix format.
        # KE = lk(E / A0, nu, A0)
        KE = lk(E, nu)
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
        fixed = union1d(dofs[0:2 * (nely + 1):2], dofs[1])
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
        # plt.ion()
        # fig, ax = plt.subplots()
        # im = ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray', interpolation='none',
        #                norm=colors.Normalize(vmin=-1, vmax=0))
        # plt.show(block=False)

        loop = 0
        change = 1
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

        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + xPhys ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = deleterowcol(K, fixed, fixed).tocoo()
        # Solve system
        K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int))
        B = cvxopt.matrix(f[free, 0])
        cvxopt.cholmod.linsolve(K, B)
        u[free, 0] = np.array(B)[:, 0]

        save = xPhys.copy()
        domain = np.zeros((xPhys.reshape((nelx, nely)).T.shape[0] + 1, xPhys.reshape((nelx, nely)).T.shape[1] + 1))
        domain[:-1, :-1] = xPhys.reshape((nelx, nely)).T
        # domain = extrapolate_domain_uniform(xPhys.reshape((nelx, nely)).T)
        input_tensor = np.stack([domain, fixed_x, fixed_y, loads_x, loads_y], axis=0)
        prediction = tester.predict_single_instance(input_tensor)
        print(prediction)
        u_x_ml = prediction[0].cpu().numpy()[0, :, :]
        u_y_ml = prediction[0].cpu().numpy()[1, :, :]
        u_ml = combine_displacement_matrices(u_x_ml, u_y_ml)

        # Objective and sensitivity
        ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)).sum(1)
        obj = ((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()

        # Calculate ML compliance energy
        # Reshape u_ml to match edofMat shape for element-wise computation
        ce_ml = np.zeros(nely * nelx)
        u_ml_reshaped = np.zeros_like(u)
        u_ml_reshaped[:len(u_ml)] = u_ml.reshape(-1, 1)
        ce_ml[:] = (np.dot(u_ml_reshaped[edofMat].reshape(nelx * nely, 8), KE) *
                    u_ml_reshaped[edofMat].reshape(nelx * nely, 8)).sum(1)
        obj_ml = ((Emin + xPhys ** penal * (Emax - Emin)) * ce_ml).sum()

        # Plot to screen
        # im.set_array(-xPhys.reshape((nelx, nely)).T)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # plt.pause(0.001)

        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
            loop, obj, (g + volfrac * nelx * nely) / (nelx * nely), change))
        print(f"ML obj.: {obj_ml:.3f}")

        # Split displacements into x and y components
        u_x, u_y = create_displacement_matrices(u, nelx, nely)

        print(f"Maximum X displacement: {np.max(np.abs(u_x))}")
        print(f"Maximum Y displacement: {np.max(np.abs(u_y))}")

        iter_group_name = f'iter_{loop}'
        if iter_group_name not in prob_group:
            iter_data = prob_group.create_group(iter_group_name)
            iter_data.create_dataset('domain', data=save.reshape((nelx, nely)).T, compression='gzip')

            # Save separated displacement components
            displacements = iter_data.create_group('displacements')
            displacements.create_dataset('x', data=u_x, compression='gzip')
            displacements.create_dataset('y', data=u_y, compression='gzip')
            displacements.create_dataset('x_ML', data=u_x_ml, compression='gzip')
            displacements.create_dataset('y_ML', data=u_y_ml, compression='gzip')

            iter_data.attrs['compliance'] = float(obj)
            iter_data.attrs['compliance_ML'] = float(obj_ml)

            # Save compliance energy
            compliance = iter_data.create_group('compliance')
            ce_reshaped = ce.reshape(nelx, nely).T
            ce_ml_reshaped = ce_ml.reshape(nelx, nely).T
            compliance.create_dataset('ce', data=ce_reshaped, compression='gzip')
            compliance.create_dataset('ce_ML', data=ce_ml_reshaped, compression='gzip')

        # Calculate normalized differences
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10

        # For displacements, we need to handle areas where the original displacement is near zero
        # We'll use absolute difference where original values are very small
        delta_u_x_norm = np.divide(u_x - u_x_ml, np.abs(u_x) + epsilon)
        delta_u_y_norm = np.divide(u_y - u_y_ml, np.abs(u_y) + epsilon)

        # For compliance energy
        ce_reshaped = ce.reshape(nelx, nely).T
        ce_ml_reshaped = ce_ml.reshape(nelx, nely).T
        delta_ce_norm = np.divide(ce_reshaped - ce_ml_reshaped, np.abs(ce_reshaped) + epsilon)

        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        dc_ml[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce_ml
        delta_dc_norm = np.divide(dc - dc_ml, np.abs(dc) + epsilon)

        # Plot displacement fields
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        im1 = axs[0].imshow(u_x, cmap='coolwarm', interpolation='none')
        axs[0].set_title("Displacement in X-direction")
        fig.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(u_y, cmap='coolwarm', interpolation='none')
        axs[1].set_title("Displacement in Y-direction")
        fig.colorbar(im2, ax=axs[1])
        fig.suptitle("SIMP displacements")

        # Plot ML displacement fields
        fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4))
        im2 = axs2[0].imshow(u_x_ml, cmap='coolwarm', interpolation='none')
        axs2[0].set_title("Displacement in X-direction")
        fig2.colorbar(im2, ax=axs2[0])

        im2 = axs2[1].imshow(u_y_ml, cmap='coolwarm', interpolation='none')
        axs2[1].set_title("Displacement in Y-direction")
        fig2.colorbar(im2, ax=axs2[1])
        fig2.suptitle("MLTO displacements")

        # Plot absolute differences
        fig3, axs3 = plt.subplots(1, 2, figsize=(10, 4))
        im3 = axs3[0].imshow(abs(u_x - u_x_ml), cmap='binary', interpolation='none')
        axs3[0].set_title("Displacement in X-direction")
        fig3.colorbar(im3, ax=axs3[0])

        im3 = axs3[1].imshow(abs(u_y - u_y_ml), cmap='binary', interpolation='none')
        axs3[1].set_title("Displacement in Y-direction")
        fig3.colorbar(im3, ax=axs3[1])
        fig3.suptitle("Absolute differences")

        # Plot normalized differences in displacement
        fig4, axs4 = plt.subplots(1, 2, figsize=(10, 4))
        # Use a symmetric colormap with limits to better visualize differences
        vmin, vmax = -1, 1  # For normalized data, -1 to 1 is often sufficient
        im4 = axs4[0].imshow(delta_u_x_norm, cmap='RdBu', interpolation='none', vmin=vmin, vmax=vmax)
        axs4[0].set_title("Normalized X-displacement Difference")
        fig4.colorbar(im4, ax=axs4[0])

        im4 = axs4[1].imshow(delta_u_y_norm, cmap='RdBu', interpolation='none', vmin=vmin, vmax=vmax)
        axs4[1].set_title("Normalized Y-displacement Difference")
        fig4.colorbar(im4, ax=axs4[1])
        fig4.suptitle("Normalized Displacement Differences (SIMP-ML)/SIMP")

        # Plot normalized differences in compliance energy
        fig5, axs5 = plt.subplots(figsize=(8, 6))
        im5 = axs5.imshow(delta_ce_norm, cmap='RdBu', interpolation='none', vmin=vmin, vmax=vmax)
        axs5.set_title("Normalized Compliance Energy Difference (SIMP-ML)/SIMP")
        fig5.colorbar(im5, ax=axs5)

        # Plot normalized differences in sensitivity
        fig6, axs6 = plt.subplots(figsize=(8, 6))
        im6 = axs6.imshow(delta_dc_norm.reshape(nelx, nely).T, cmap='RdBu', interpolation='none', vmin=vmin, vmax=vmax)
        axs6.set_title("Normalized Sensitivity Difference (SIMP-ML)/SIMP")
        fig6.colorbar(im6, ax=axs6)

        plt.tight_layout()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14

        fig7, axs7 = plt.subplots(figsize=(6, 2))
        im7 = axs7.imshow(dc.reshape((nelx, nely)).T, cmap='inferno', interpolation='none')
        fig7.colorbar(im7, ax=axs7, fraction=0.046, pad=0.04, aspect=10)

        fig8, axs8 = plt.subplots(figsize=(6, 2))
        im8 = axs8.imshow(dc_ml.reshape((nelx, nely)).T, cmap='inferno', interpolation='none')
        fig8.colorbar(im8, ax=axs8, fraction=0.046, pad=0.04, aspect=10)

        # Save all figures
        fig.savefig("01_FEM_SIMP_displacements.png")
        fig2.savefig("02_FEM_MLTO_displacements.png")
        fig3.savefig("03_FEM_Diff_displacements.png")
        fig4.savefig("04_FEM_Norm_Diff_displacements.png")
        fig5.savefig("05_FEM_Norm_Diff_compliance.png")
        fig6.savefig("06_FEM_Norm_Diff_sensitivity.png")
        fig7.savefig("dc.svg", format='svg', bbox_inches='tight')
        fig8.savefig("dc_ml_loss_bcs.svg", format='svg', bbox_inches='tight')

        plt.pause(0.001)
        plt.tight_layout()
        # plt.show(block=False)
        # input("Press any key...")

        # Save final results
        if 'results' not in prob_group:
            results = prob_group.create_group('results')
            results.create_dataset('final_domain', data=xPhys.reshape((nelx, nely)).T, compression='gzip')
            results.attrs['final_compliance'] = float(obj)
            results.attrs['final_compliance_ML'] = float(obj_ml)
            results.attrs['total_iterations'] = loop

    finally:
        h5file.close()

    return xPhys, obj


def lk(E, nu):
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


def deleterowcol(A, delrow, delcol):
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A


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


def create_displacement_matrices(displacement_vector, nelx, nely):
    # Create empty matrices for x and y displacements
    x_disps = np.zeros(((nelx + 1), (nely + 1)))
    y_disps = np.zeros(((nelx + 1), (nely + 1)))

    # Process all DOFs
    for dof in range(len(displacement_vector)):
        # Even indices are x displacements, odd indices are y displacements
        is_y_disp = dof % 2
        # Convert DOF index to node index
        node_index = dof // 2

        # Get x, y coordinates from node index
        x_coord = node_index // (nely + 1)
        y_coord = node_index % (nely + 1)

        # Set displacement value at appropriate position
        if is_y_disp:
            y_disps[x_coord, y_coord] = displacement_vector[dof].item()
        else:
            x_disps[x_coord, y_coord] = displacement_vector[dof].item()

    return x_disps.T, y_disps.T


def combine_displacement_matrices(u_x, u_y):
    # Transpose back to match original order if needed
    u_x = u_x.T
    u_y = u_y.T

    # Flatten the displacement matrices
    u_x_flat = u_x.flatten()
    u_y_flat = u_y.flatten()

    # Interleave x and y displacements
    u = np.zeros(2 * len(u_x_flat))
    u[0::2] = u_x_flat  # Even indices store x-displacements
    u[1::2] = u_y_flat  # Odd indices store y-displacements

    return u


def extrapolate_domain_uniform(domain):
    """
    Extrapolate element-wise domain to nodal grid using uniform averaging.
    domain: (H, W) numpy array of element-wise densities
    returns: (H+1, W+1) numpy array of node-wise densities
    """
    H, W = domain.shape
    nodal = np.zeros((H+1, W+1))
    counts = np.zeros((H+1, W+1))

    for i in range(H):
        for j in range(W):
            e_density = domain[i, j]
            # Share equally to the 4 surrounding nodes
            for dy in [0, 1]:
                for dx in [0, 1]:
                    n_y = i + dy
                    n_x = j + dx
                    nodal[n_y, n_x] += e_density
                    counts[n_y, n_x] += 1

    nodal /= np.maximum(counts, 1e-8)
    return nodal


if __name__ == "__main__":
    # Default input parameters
    volfrac = 0.5
    rmin = 5.4
    penal = 3.0
    nelx = 180
    nely = 60

    model_path = '../CNN-model/models/topology_Unet_model_Loss_BC.pkl'
    stats_path = '../CNN-model/dataset_stats_loss_bcs.json'

    # Load configuration
    load_config = {
        'position': 1,
        'horizontal_magnitude': 0,
        'vertical_magnitude': 50
    }

    xPhys, obj = fea(nelx, nely, volfrac, load_config, penal, model_path, stats_path)

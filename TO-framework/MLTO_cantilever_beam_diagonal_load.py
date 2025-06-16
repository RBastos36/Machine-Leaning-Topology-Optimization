# A 200 line Topology Optimization code by Niels Aaage and Villads Egede Johansen, January 2013
# Updated by Niels Aage, February 2016
# Adapted by Ricardo Bastos, June 2025


from __future__ import division
import math
from scipy.sparse import coo_matrix
from matplotlib import colors

# Import U-Net model
from ML_framework import *


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


def topopt(nelx, nely, volfrac, penal, rmin, ft, load_config):
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])
    print(f"Load config: {load_config}")

    # Initialize dataset if needed
    initialize_dataset('cantilever_diagonal_framework.h5')

    # Open the HDF5 file at the start and keep it open
    h5file = h5py.File('cantilever_diagonal_framework.h5', 'a')

    # ML initializations
    model_path = '../CNN-model/models/topology_Unet_model_ORIGINAL.pkl'
    device = None
    with open('../CNN-model/dataset_stats.json', 'r') as f:
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

        # dofs
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

        loop = 0
        change = 1
        dv = np.ones(nely * nelx)
        dc = np.ones(nely * nelx)
        ce = np.ones(nely * nelx)

        # Initialize plot
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray', interpolation='none',
                       norm=colors.Normalize(vmin=-1, vmax=0))
        plt.show(block=False)

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
            # Substitute FEM displacements by the U-Net model
            xPhys_2_save = xPhys
            if loop == 1:
                fig.savefig(f"z_initial_domain_0.8.svg", bbox_inches='tight')
                break
            domain = np.zeros((xPhys.reshape((nelx, nely)).T.shape[0] + 1, xPhys.reshape((nelx, nely)).T.shape[1] + 1))
            domain[:-1, :-1] = xPhys.reshape((nelx, nely)).T
            # domain = extrapolate_domain_uniform(xPhys.reshape((nelx, nely)).T)
            input_tensor = np.stack([domain, loads_x, loads_y, fixed_x, fixed_y], axis=0)
            prediction = tester.predict_single_instance(input_tensor)
            u_x = prediction[0].cpu().numpy()[0, :, :]
            u_y = prediction[0].cpu().numpy()[1, :, :]
            u = combine_displacement_matrices(u_x, u_y)

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

            if loop == 1 or loop == 10 or loop == 100 or loop == 1000:
                fig.savefig(f"iter_{loop}_loss_bcs.svg", bbox_inches='tight')

            # Save iteration data periodically
            if loop <= 10 or 2 ** (int(math.log2(loop))) == loop:
                iter_group_name = f'iter_{loop}'
                if iter_group_name not in prob_group:
                    iter_data = prob_group.create_group(iter_group_name)
                    iter_data.create_dataset('domain', data=xPhys_2_save.reshape((nelx, nely)).T, compression='gzip')

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


if __name__ == "__main__":
    # Default input parameters
    nelx = 180
    nely = 60
    volfrac = 0.8
    rmin = 5.4
    penal = 3.0
    ft = 0  # ft==0 -> sens, ft==1 -> dens

    load_config = {
        'position': 1,
        'horizontal_magnitude': 0,
        'vertical_magnitude': 50
    }

    xPhys, obj = topopt(nelx, nely, volfrac, penal, rmin, ft, load_config)

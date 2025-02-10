# A 200 line Topology Optimization code by Niels Aaage and Villads Egede Johansen, January 2013
# Updated by Niels Aage, February 2016
# Adapted by Ricardo Bastos, January 2025

from __future__ import division
import numpy as np
from numpy.ma.extras import union1d
from scipy.sparse import coo_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.cholmod
import time
import h5py


# start_time = time.time()


def topopt(nelx, nely, volfrac, penal, rmin, ft, load_config, void_start_x_frac, void_end_y_frac,
           void_end_x_frac_2, void_start_y_frac_2):
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])
    print(f"Load config: {load_config}")
    print(f"Void region fractions: {void_start_x_frac} (x_start), {void_end_y_frac} (y_end)")
    print(f"Void region 2 fractions: {void_end_x_frac_2} (x_end), {void_start_y_frac_2} (y_start)")


    # Create void region mask (1 for material allowed, 0 for void)
    material_mask = np.ones((nely, nelx))
    material_mask[int(nelx * void_start_x_frac):, :int(nely * void_end_y_frac)] = 0  # Set top right region to void
    material_mask[:int(nelx * void_end_x_frac_2), int(nely * void_start_y_frac_2):] = 0  # Set top right region to void


    # Initialize dataset if needed
    initialize_dataset('l-bracket-modified_dataset.h5')

    # Open the HDF5 file at the start and keep it open
    h5file = h5py.File('l-bracket-modified_dataset.h5', 'a')

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
            params.attrs['load_direction'] = load_config['direction']
            params.attrs['load_magnitude'] = load_config['magnitude']
            params.attrs['void_start_x_frac'] = void_start_x_frac
            params.attrs['void_end_y_frac'] = void_end_y_frac
            params.attrs['void_end_x_frac_2'] = void_end_x_frac_2
            params.attrs['void_start_y_frac_2'] = void_start_y_frac_2

        # Max and min stiffness
        Emin = 1e-9
        Emax = 1.0

        # dofs:
        ndof = 2 * (nelx + 1) * (nely + 1)

        # Allocate design variables (as array), initialize and allocate sens.
        x_0 = np.ones(nely * nelx, dtype=float)
        x = volfrac * x_0
        x = x * material_mask.flatten()
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

        # BC's and support (top edge fixed)
        dofs = np.arange(2 * (nelx + 1) * (nely + 1))
        fixed = dofs[0:2 * (nely + 1):1]  # Fix all DOFs on left edge
        free = np.setdiff1d(dofs, fixed)

        # Set up load vector
        f = np.zeros((ndof, 1))

        # Calculate load position
        rel_position = load_config['position']  # Between 0 and 1
        void_end_y = int(nely * void_end_y_frac)  # Extend void to _% of height
        short_edge_length = nely - void_end_y  # Length of the shorter right edge
        node_y_pos = int(rel_position * short_edge_length)
        node_index = nelx * (nely + 1) + void_end_y + node_y_pos

        # Set load direction and magnitude
        if load_config['direction'] == 'horizontal':
            # For horizontal load
            dof = 2 * node_index  # x-direction DOF
        else:  # vertical
            dof = 2 * node_index + 1  # y-direction DOF

        f[dof, 0] = load_config['magnitude']

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

            # Objective and sensitivity
            ce[:] = (np.dot(u[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)).sum(1)
            obj = ((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()
            dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
            dc = dc * material_mask.flatten()  # Apply material mask to sensitivities

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

            # Apply material mask after optimization step
            x[:] = x * material_mask.flatten()

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

            # Split displacements into x and y components
            u_x, u_y = create_displacement_matrices(u, nelx, nely)

            # Save iteration data periodically
            if loop % 10 == 0:
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
        # input("Press any key...")

        # Save final results
        if 'results' not in prob_group:
            results = prob_group.create_group('results')
            results.create_dataset('final_domain', data=xPhys.reshape((nelx, nely)).T, compression='gzip')
            results.attrs['final_compliance'] = float(obj)
            results.attrs['total_iterations'] = loop

    finally:
        h5file.close()

    return xPhys, obj


# Rest of the functions remain the same
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
            f"dir{load_config['direction'][0]}_"  # First letter of direction
            f"mag{abs(load_config['magnitude'])}_"
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


if __name__ == "__main__":
    # Default input parameters
    nelx = 100
    nely = 100
    volfrac = 0.4
    rmin = 5.4
    penal = 3.0
    ft = 0  # ft==0 -> sens, ft==1 -> dens
    void_start_x_frac = 0.5
    void_end_y_frac = 0.5
    void_end_x_frac_2 = 0.2
    void_start_y_frac_2 = 0.2

    # Load configuration
    load_config = {
        'position': 0.5,  # 1 - bottom of short right edge / 0 - top of short right edge
        'direction': 'vertical',  # 'vertical' or 'horizontal'
        'magnitude': -1.0  # negative for downward/leftward force
    }

    xPhys, obj = topopt(nelx, nely, volfrac, penal, rmin, ft, load_config, void_start_x_frac, void_end_y_frac,
                        void_end_x_frac_2, void_start_y_frac_2)

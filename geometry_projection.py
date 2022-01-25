import numpy as np
import scipy.sparse as sp

# from LP_routines_v1 import *
from FE_routines import *


def init_geometry(FE, OPT, GEOM):
    #
    # Initialize GEOM structure with initial design
    #

    if not GEOM['initial_design']['restart']:
        exec(open(GEOM['initial_design']['path']).read())

        # To use non contiguous numbers in the point_mat, we need to grab the
        # points whose ID matches the number specified by bar_mat. We achieve
        # this via a map (sparse vector) between point_mat_rows and pt_IDs st
        # point_mat_row(point_ID) = row # of point_mat for point_ID
        pt_IDs = GEOM['initial_design']['point_matrix'][:, 0]

        GEOM['point_mat_row'] = sp.csc_matrix((np.arange(0, pt_IDs.shape[0]),
                                               (pt_IDs.astype(int), np.zeros(pt_IDs.shape[0], dtype=int))))
    else:
        exec(open(GEOM['initial_design']['path']).read())
        GEOM['initial_design']['point_matrix'] = GEOM['current_design']['point_matrix']
        GEOM['initial_design']['bar_matrix'] = GEOM['current_design']['bar_matrix']

    GEOM['n_point'] = np.size(GEOM['initial_design']['point_matrix'], 0)
    GEOM['n_bar'] = np.size(GEOM['initial_design']['bar_matrix'], 0)


def compute_bar_element_stiffness(FE, C):
    C11 = np.squeeze(C[0, 0, :])
    C12 = np.squeeze(C[0, 1, :])
    C13 = np.squeeze(C[0, 2, :])
    C22 = np.squeeze(C[1, 1, :])
    C23 = np.squeeze(C[1, 2, :])
    C33 = np.squeeze(C[2, 2, :])

    Ke = np.zeros((8, 8, C.shape[2]))

    Delta1 = 1
    Delta2 = 1

    Ke[1, 0, :] = C12/4+C13*Delta2/(3*Delta1)+C23*Delta1/(3*Delta2)+C33/4
    Ke[2, 0, :] = -C11*Delta2/(3*Delta1)+C33*Delta1/(6*Delta2)
    Ke[3, 0, :] = C12/4-C13*Delta2/(3*Delta1)+C23*Delta1/(6*Delta2)-C33/4
    Ke[4, 0, :] = -C11*Delta2/(6*Delta1)-C13/2-C33*Delta1/(6*Delta2)
    Ke[5, 0, :] = -C12/4-C13*Delta2/(6*Delta1)-C23*Delta1/(6*Delta2)-C33/4
    Ke[6, 0, :] = C11*Delta2/(6*Delta1)-C33*Delta1/(3*Delta2)
    Ke[7, 0, :] = -C12/4+C13*Delta2/(6*Delta1)-C23*Delta1/(3*Delta2)+C33/4
    Ke[2, 1, :] = -C12/4-C13*Delta2/(3*Delta1)+C23*Delta1/(6*Delta2)+C33/4
    Ke[3, 1, :] = C22*Delta1/(6*Delta2)-C33*Delta2/(3*Delta1)
    Ke[4, 1, :] = -C12/4-C13*Delta2/(6*Delta1)-C23*Delta1/(6*Delta2)-C33/4
    Ke[5, 1, :] = -C22*Delta1/(6*Delta2)-C23/2-C33*Delta2/(6*Delta1)
    Ke[6, 1, :] = C12/4+C13*Delta2/(6*Delta1)-C23*Delta1/(3*Delta2)-C33/4
    Ke[7, 1, :] = -C22*Delta1/(3*Delta2)+C33*Delta2/(6*Delta1)
    Ke[3, 2, :] = -C12/4+C13*Delta2/(3*Delta1)+C23*Delta1/(3*Delta2)-C33/4
    Ke[4, 2, :] = C11*Delta2/(6*Delta1)-C33*Delta1/(3*Delta2)
    Ke[5, 2, :] = C12/4+C13*Delta2/(6*Delta1)-C23*Delta1/(3*Delta2)-C33/4
    Ke[6, 2, :] = -C11*Delta2/(6*Delta1)+C13/2-C33*Delta1/(6*Delta2)
    Ke[7, 2, :] = C12/4-C13*Delta2/(6*Delta1)-C23*Delta1/(6*Delta2)+C33/4
    Ke[4, 3, :] = -C12/4+C13*Delta2/(6*Delta1)-C23*Delta1/(3*Delta2)+C33/4
    Ke[5, 3, :] = -C22*Delta1/(3*Delta2)+C33*Delta2/(6*Delta1)
    Ke[6, 3, :] = C12/4-C13*Delta2/(6*Delta1)-C23*Delta1/(6*Delta2)+C33/4
    Ke[7, 3, :] = -C22*Delta1/(6*Delta2)+C23/2-C33*Delta2/(6*Delta1)
    Ke[5, 4, :] = C12/4+C13*Delta2/(3*Delta1)+C23*Delta1/(3*Delta2)+C33/4
    Ke[6, 4, :] = -C11*Delta2/(3*Delta1)+C33*Delta1/(6*Delta2)
    Ke[7, 4, :] = C12/4-C13*Delta2/(3*Delta1)+C23*Delta1/(6*Delta2)-C33/4
    Ke[6, 5, :] = -C12/4-C13*Delta2/(3*Delta1)+C23*Delta1/(6*Delta2)+C33/4
    Ke[7, 5, :] = C22*Delta1/(6*Delta2)-C33*Delta2/(3*Delta1)
    Ke[7, 6, :] = -C12/4+C13*Delta2/(3*Delta1)+C23*Delta1/(3*Delta2)-C33/4

    Ke = Ke + Ke.transpose(1, 0, 2)

    Ke[0, 0, :] = C11*Delta2/(3*Delta1)+C13/2+C33*Delta1/(3*Delta2)
    Ke[1, 1, :] = C22*Delta1/(3*Delta2)+C23/2+C33*Delta2/(3*Delta1)
    Ke[2, 2, :] = C11*Delta2/(3*Delta1)-C13/2+C33*Delta1/(3*Delta2)
    Ke[3, 3, :] = C22*Delta1/(3*Delta2)-C23/2+C33*Delta2/(3*Delta1)
    Ke[4, 4, :] = C11*Delta2/(3*Delta1)+C13/2+C33*Delta1/(3*Delta2)
    Ke[5, 5, :] = C22*Delta1/(3*Delta2)+C23/2+C33*Delta2/(3*Delta1)
    Ke[6, 6, :] = C11*Delta2/(3*Delta1)-C13/2+C33*Delta1/(3.*Delta2)
    Ke[7, 7, :] = C22*Delta1/(3*Delta2)-C23/2+C33*Delta2/(3*Delta1)

    return Ke


def compute_bar_elem_distance(FE, OPT, GEOM):
    # global FE, GEOM, OPT

    tol = 1e-12

    n_elem = FE['n_elem']
    dim = FE['dim']
    n_bar = GEOM['n_bar']
    n_bar_dof = 2*dim

    # (dim,bar,elem)
    points = GEOM['current_design']['point_matrix'][:, 1:].T

    x_1b = points.T.flatten()[OPT['bar_dv'][0:dim, :]]  # (i,b)
    x_2b = points.T.flatten()[OPT['bar_dv'][dim:2*dim, :]]  # (i,b)
    x_e = FE['centroids']                        # (i,1,e)

    a_b = x_2b - x_1b
    l_b = np.sqrt(np.sum(a_b**2, 0))  # length of the bars, Eq. (10)
    l_b[np.where(l_b < tol)] = 1          # To avoid division by zero
    # normalize the bar direction to unit vector, Eq. (11)
    a_b = np.divide(a_b, l_b)

    x_e_1b = (x_e.T[:, None] - x_1b.T).swapaxes(0, 2)               # (i,b,e)
    x_e_2b = (x_e.T[:, None] - x_2b.T).swapaxes(0, 2)                 # (i,b,e)
    norm_x_e_1b = np.sqrt(np.sum(x_e_1b**2, 0))  # (1,b,e)
    norm_x_e_2b = np.sqrt(np.sum(x_e_2b**2, 0))   # (1,b,e)

    # (1,b,e), Eq. (12)
    l_be = np.sum(x_e_1b * a_b[:, :, None], 0)
    vec_r_be = x_e_1b - (l_be.T * a_b[:, None]).swapaxes(1, 2)      # (i,b,e)
    r_be = np.sqrt(np.sum(vec_r_be**2, 0))    # (1,b,e), Eq. (13)

    l_be_over_l_b = (l_be.T / l_b).T

    branch1 = l_be <= 0.0   # (1,b,e)
    branch2 = l_be_over_l_b >= 1   # (1,b,e)
    branch3 = np.logical_not(np.logical_or(branch1, branch2))    # (1,b,e)

    # Compute the distances, Eq. (14)
    dist = branch1 * norm_x_e_1b + \
        branch2 * norm_x_e_2b + \
        branch3 * r_be         # (1,b,e)

    # compute sensitivities
    Dd_be_Dx_1b = np.zeros((FE['dim'], n_bar, n_elem))
    Dd_be_Dx_2b = np.zeros((FE['dim'], n_bar, n_elem))

    d_inv = dist**(-1)           # This can rer a division by zero
    d_inv[np.isinf(d_inv)] = 0  # lies on medial axis, and so we now fix it

    # The sensitivities below are obtained from Eq. (30)
    # sensitivity to x_1b
    Dd_be_Dx_1b = -x_e_1b * d_inv * branch1 + \
        -vec_r_be * d_inv * (1 - l_be_over_l_b) * branch3

    Dd_be_Dx_2b = -x_e_2b * d_inv * branch2 + \
        -vec_r_be * d_inv * l_be_over_l_b * branch3

    # assemble the sensitivities to the bar design parameters (scaled)
    Dd_be_Dbar_ends = np.concatenate((Dd_be_Dx_1b, Dd_be_Dx_2b),
                                     axis=0).transpose((1, 2, 0)) * \
        np.concatenate((OPT['scaling']['point_scale'],
                       OPT['scaling']['point_scale']))
    # print( Dd_be_Dx_1b[:,1000:1005].transpose((2,0,1)) )
    # time.sleep(10)

    return dist, Dd_be_Dbar_ends


def penalize(*args):

    # [P, dPdx] = penalize(x, p, penal_scheme)
    #     penalize(x) assumes x \in [0,1] and decreases the intermediate values
    #
    #	  For a single input, the interpolation is SIMP with p = 3
    #
    #	  The optional second argument is the parameter value p.
    #
    #     The optional third argument is a string that indicates the way the
    #	  interpolation is defined, possible values are:
    #       'SIMP'      : default
    # 	  	'RAMP'      :
    #

    # consider input
    n_inputs = len(args)
    x = args[0]
    if n_inputs == 1:
        # set the definition to be used by default.
        p = 3
        penal_scheme = 'SIMP'
    elif n_inputs == 2:
        p = args[1]
        penal_scheme = 'SIMP'
    elif n_inputs == 3:
        p = args[1]
        penal_scheme = args[2]

    # consider output
    # not implemented

    # define def
    if penal_scheme == 'SIMP':
        P = x**p
        dPdx = p * x**(p-1)
    elif penal_scheme == 'RAMP':
        P = x / (1 + p*(1-x))
        dPdx = (1+p) / (1 + p*(1-x))**2
    else:
        print('Unidentified parameters')

    # compute the output
    return P, dPdx


def compute(FE, OPT, GEOM):
    # set paramters
    tol = 1e-12

    dim = FE['dim']
    n_bar = GEOM['n_bar']

    points = GEOM['current_design']['point_matrix'][:, 1:].T


def compute_bar_basis(FE, GEOM, x_sb):

    dim = FE['dim']
    n_bar = GEOM['n_bar']
    bar_tol = 1e-12

    # e_1b
    # Princple direction along bar axis
    v_b = np.zeros((3, n_bar))
    v_b[:dim, :] = x_sb[:, 1, :] - x_sb[:, 0, :]
    norm_v_b = np.maximum(np.sqrt(np.sum(v_b*v_b, 0)), bar_tol)
    e_1b = (v_b/norm_v_b)[:, :, None]             # (3,b,1)

    P_1b_perp = np.zeros((3, 3, n_bar))
    P_1b_perp = np.repeat(np.eye(
        3)[:, :, None], n_bar, axis=2) - e_1b.transpose(0, 2, 1)*e_1b.transpose(2, 0, 1)
    norm_v_b_perm = norm_v_b[:, None, None].transpose(2, 1, 0)
    De_1b_Dx_2b = np.zeros((3, dim, n_bar))
    De_1b_Dx_2b = P_1b_perp[:, :dim, :]/norm_v_b_perm

    # e_2b
    # determine coordinate direction most orthogonal to bar
    e_alpha = np.zeros((3, n_bar))

    case_1 = (abs(v_b[0, :]) < abs(v_b[1, :])) & (
        abs(v_b[0, :]) < abs(v_b[2, :]))
    case_2 = (abs(v_b[1, :]) < abs(v_b[0, :])) & (
        abs(v_b[1, :]) < abs(v_b[2, :]))
    case_3 = np.logical_not(case_1 | case_2)

    e_alpha[0, case_1] = 1
    e_alpha[1, case_2] = 1
    e_alpha[2, case_3] = 1

    # cross product are defined along the first axis, rather than the last axis. these axes can have dimensions 2 or 3. the last axis is only of length 1, so the cross product is not defined.
    v_2b = np.cross(e_alpha, np.squeeze(e_1b), axis=0)
    norm_v_2b = np.maximum(np.sqrt(np.sum(v_2b**2, 0)), bar_tol)
    e_2b = (v_2b/norm_v_2b)[:, :, None]

    epsilon = np.zeros((3, 3, 3))

    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[1, 0, 2] = epsilon[0, 2, 1] = -1

    alpha = np.sum(e_alpha * np.arange(3)[:, None], dtype=int, axis=0)

    eps_alpha_DOT_De_1b_Dx_sb = np.zeros((dim, 3, n_bar))
    for nbar, val in zip(np.arange(n_bar), alpha):
        eps_alpha_DOT_De_1b_Dx_sb[:, :, nbar] = np.sum(
            epsilon[:, None, :, val]*De_1b_Dx_2b[:, :, np.newaxis, nbar], axis=0)

    P_2b_perp = np.zeros((3, 3, n_bar))
    P_2b_perp = np.repeat(np.eye(
        3)[:, :, None], n_bar, axis=2) - e_2b.transpose(0, 2, 1)*e_2b.transpose(2, 0, 1)

    De_2b_Dx_2b = np.zeros((3, dim, n_bar))
    De_2b_Dx_2b = np.sum(
        eps_alpha_DOT_De_1b_Dx_sb[None, :, :, :]*P_2b_perp[:, None, :, :], axis=2) / norm_v_2b[None, None, :]

    # e_3b
    v_3b = np.cross(np.squeeze(e_2b), np.squeeze(e_1b), axis=0)
    norm_v_3b = np.maximum(np.sqrt(np.sum(v_3b**2, 0)), bar_tol)
    e_3b = (v_3b/norm_v_3b)[:, :, None]

    De_3b_Dx_2b = np.zeros((3, dim, n_bar))

    return np.squeeze(e_1b), np.squeeze(e_2b), np.squeeze(e_3b), De_1b_Dx_2b, De_2b_Dx_2b


def compute_bar_material_orientation(FE, GEOM):
    bar_tol = 1e-12
    dim = FE['dim']
    n_bar = GEOM['n_bar']

    point_mat = GEOM['current_design']['point_matrix']
    bar_mat = GEOM['current_design']['bar_matrix']

    pt1_IDs = bar_mat[:, 1].astype(int)
    pt2_IDs = bar_mat[:, 2].astype(int)

    x_sb = np.zeros((dim, 2, n_bar))
    x_sb[0:dim, 0, :] = point_mat[GEOM['point_mat_row'][pt1_IDs].toarray()[
        :, 0], 1:].T
    x_sb[0:dim, 1, :] = point_mat[GEOM['point_mat_row'][pt2_IDs].toarray()[
        :, 0], 1:].T

    e_hat_1b, e_hat_2b, e_hat_3b, De_hat_1b_Dx_2b, De_hat_2b_Dx_2b = compute_bar_basis(
        FE, GEOM, x_sb)

    FE['e_hat_1b'] = e_hat_1b

    # Jacobian transformation (rotation) matrix R
    R_b = np.zeros((3, 3, n_bar))
    R_b[:, 0, :] = e_hat_1b
    R_b[:, 1, :] = e_hat_2b
    R_b[:, 2, :] = e_hat_3b

    DR_b_Dx_2b = np.zeros((3, 3, n_bar, dim))
    DR_b_Dx_2b[:, 0, :, :] = De_hat_1b_Dx_2b.transpose(0, 2, 1)
    DR_b_Dx_2b[:, 1, :, :] = De_hat_2b_Dx_2b.transpose(0, 2, 1)

    # Rotation in Voigt Notation

    I = np.array([0, 1, 2])  # 11, 22, 33
    J = np.array([3, 4, 5])  # 23, 13, 12
    i = np.array([1, 0, 0])
    j = np.array([2, 2, 1])

    RR_b = np.zeros((6, 6, n_bar))
    RR_b[I, I, :] = R_b[I, I, :]*R_b[I, I, :]
    RR_b[J, I, :] = R_b[i, I, :]*R_b[j, I, :]
    RR_b[I, J, :] = 2*R_b[I, i, :]*R_b[I, j, :]
    RR_b[J, J, :] = R_b[i, i, :]*R_b[j, j, :] + R_b[i, j, :]*R_b[j, i, :]

    DRR_b_Dn_b = np.zeros((6, 6, n_bar, dim))

    DRR_b_Dn_b[I, I, :, :] \
        = 2*R_b[I, I, :, None] * DR_b_Dx_2b[I, I, :, :]

    DRR_b_Dn_b[J, I, :, :] \
        = R_b[i, I, :, None] * DR_b_Dx_2b[j, I, :, :] \
        + DR_b_Dx_2b[i, I, :, :]*R_b[j, I, :, None]

    DRR_b_Dn_b[I, J, :, :] \
        = 2*R_b[I, i, :, None]*DR_b_Dx_2b[I, j, :, :] \
        + 2*DR_b_Dx_2b[I, i, :, :]*R_b[I, j, :, None]

    DRR_b_Dn_b[J, J, :, :] \
        = R_b[i, i, :, None]*DR_b_Dx_2b[j, j, :, :] \
        + DR_b_Dx_2b[i, i, :, :]*R_b[j, j, :, None] \
        + R_b[i, j, :, None]*DR_b_Dx_2b[j, i, :, :] \
        + DR_b_Dx_2b[i, j, :, :] * R_b[j, i, :, None]

    if FE['dim'] == 2:
        RR_b = RR_b[[0, 1, 5], :, :][:, [0, 1, 5], :]
        DRR_b_Dn_b = DRR_b_Dn_b[[0, 1, 5], :, :][:, [0, 1, 5], :, :]

    FE['RR_b'] = RR_b
    FE['DRR_b_Dn_b'] = DRR_b_Dn_b

    FE['material']['Cb'] = np.zeros((3, 3, GEOM['n_bar']))
    FE['material']['DCb_Dbar_vector'] = np.zeros((3, 3, n_bar, dim))

    C0 = FE['material']['Ql']
    point_scale = FE['mesh_input']['box_dimensions']
    for nbar in range(n_bar):

        C = np.dot(np.dot(RR_b[:, :, nbar], C0),
                   np.transpose(RR_b[:, :, nbar]))

        DC_Dbar_vec = np.zeros((3, 3, dim))
        for d in range(dim):
            DC_Dbar_vec[:, :, d] = DRR_b_Dn_b[:, :, nbar, d]*C0*RR_b[:, :, nbar].T + \
                RR_b[:, :, nbar]*C0*DRR_b_Dn_b[:, :, nbar, d] * \
                DRR_b_Dn_b[:, :, nbar, d].T

        FE['material']['Cb'][:, :, nbar] = C


def project_element_densities(FE, OPT, GEOM):
    # This def computes the combined unpenalized densities (used to
    # compute the volume) and penalized densities (used to compute the ersatz
    # material for the analysis) and saves them in the global variables
    # FE['elem_dens'] and FE['penalized_elem_dens'].
    #
    # It also computes the derivatives of the unpenalized and penalized
    # densities with respect to the design parameters, and saves them in the
    # global variables FE['Delem_dens_Ddv'] and FE['Dpenalized_elem_dens_Ddv'].
    #

    dim = FE['dim']
    n_bar = GEOM['n_bar']
    n_dv = OPT['n_dv']
    n_bar_dv = OPT['bar_dv'].shape[0]
    bar_ends = np.arange(2*dim)
    bar_radii = bar_ends[-1] + 1
    bar_size = bar_radii + 1

    # Number of elements and nodes.
    elements_per_side = FE['mesh_input']['elements_per_side']
    FE['n_elem'] = np.prod(elements_per_side[:])
    FE['n_node'] = np.prod(elements_per_side[:]+1)

    # Distances from the element centroids to the medial segment of each bar
    d_be, Dd_be_Dbar_ends = compute_bar_elem_distance(FE, OPT, GEOM)

    # Bar-element projected densities
    r_b = GEOM['current_design']['bar_matrix'][:, -1]  # bar radii
    r_e = OPT['parameters']['elem_r']  # sample window radius

    # X_be is \phi_b/r in Eq. (2).  Note that the numerator corresponds to
    # the signed distance of Eq. (8).
    X_be = (r_b[:, None] - d_be) / r_e[None, :]

    FE['bar_element_bool'] = X_be > -1

    b_ind, e_ind = np.where(FE['bar_element_bool'])
    n_be = len(b_ind)

    E = np.unique(e_ind)
    n_E = len(E)
    E_ind = sp.csr_matrix(
        (E, (np.arange(0, len(E)), np.zeros(len(E), dtype=int))))

    be_ind_list = [[]]*GEOM['n_bar']
    e_set_list = [[]]*GEOM['n_bar']
    E_set_list = [[]]*GEOM['n_bar']
    for b in range(GEOM['n_bar']):
        be_ind_list[b] = np.array(np.where(b_ind == b)).flatten()
        e_set_list[b] = e_ind[be_ind_list[b]].reshape(be_ind_list[b].size, 1)
        E_set_list[b] = np.searchsorted(
            E_ind.toarray().flatten(), np.intersect1d(E_ind.toarray(), e_set_list[b]))

    # Projected density
    # Initial definitions
    rho_be = np.zeros((GEOM['n_bar'], FE['n_elem']))
    Drho_be_Dx_be = np.zeros((GEOM['n_bar'], FE['n_elem']))
    # In the boundary
    inB = np.abs(X_be) < 1
    # Inside
    ins = 1 <= X_be
    rho_be[ins] = 1

    if FE['dim'] == 2:  # 2D
        rho_be[inB] = 1 + (X_be[inB]*np.sqrt(1.0 - X_be[inB]
                           ** 2) - np.arccos(X_be[inB])) / np.pi
        Drho_be_Dx_be[inB] = (np.sqrt(1.0 - X_be[inB]**2)
                              * 2.0) / np.pi  # Eq. (28)
        # rho_be = np.arctan(3*X_be)/np.pi + 0.5
        # Drho_be_Dx_be = 3/(np.pi*(1+9*X_be**2))
    elif FE['dim'] == 3:
        rho_be[inB] = ((X_be[inB]-2.0)*(-1.0/4.0)*(X_be[inB]+1.0)**2)
        Drho_be_Dx_be[inB] = (X_be[inB]**2*(-3.0/4.0)+3.0/4.0)  # Eq. (28)

    # Sensitivities of raw projected densities, Eqs. (27) and (29)
    Drho_be_Dbar_ends = (Drho_be_Dx_be * -1/r_e *
                         Dd_be_Dbar_ends.transpose((2, 0, 1))).transpose((1, 2, 0))

    Drho_be_Dbar_radii = OPT['scaling']['radius_scale'] * \
        Drho_be_Dx_be * np.transpose(1/r_e)

    # Combined densities
    # Get size variables
    alpha_b = GEOM['current_design']['bar_matrix'][:, -2]  # bar size

    # Without penalization:
    # ====================
    # X_be here is \hat{\rho}_b in Eq. (4) with the value of q such that
    # there is no penalization (e.g., q = 1 in SIMP).
    X_be = alpha_b[:, None] * rho_be

    # Sensitivities of unpenalized effective densities, Eq. (26) with
    # ?\partial \mu / \partial (\alpha_b \rho_{be})=1
    DX_be_Dbar_s = Drho_be_Dbar_ends * alpha_b[:, None, None]
    DX_be_Dbar_size = rho_be.copy()
    DX_be_Dbar_radii = Drho_be_Dbar_radii * alpha_b[:, None]

    # Combined density of Eq. (5).
    rho_e, Drho_e_DX_be = smooth_max(X_be,
                                     OPT['parameters']['smooth_max_param'],
                                     OPT['parameters']['smooth_max_scheme'],
                                     FE['material']['rho_min'])

    # Sensitivities of combined densities, Eq. (25)
    Drho_e_Dbar_s = Drho_e_DX_be[:, :, None] * DX_be_Dbar_s
    Drho_e_Dbar_size = Drho_e_DX_be * DX_be_Dbar_size
    Drho_e_Dbar_radii = Drho_e_DX_be * DX_be_Dbar_radii

    # Stack together sensitivities with respect to different design
    # variables into a single vector per element
    Drho_e_Ddv = np.zeros((FE['n_elem'], OPT['n_dv']))
    for b in range(0, GEOM['n_bar']):
        Drho_e_Ddv[:, OPT['bar_dv'][:, b]] = \
            Drho_e_Ddv[:, OPT['bar_dv'][:, b]] + np.concatenate((
                Drho_e_Dbar_s[b, :, :].reshape(
                    (FE['n_elem'], 2*FE['dim']), order='F'),
                Drho_e_Dbar_size[b, :].reshape((FE['n_elem'], 1)),
                Drho_e_Dbar_radii[b, :].reshape((FE['n_elem'], 1))), axis=1)

    # With penalization:
    # =================
    # In this case X_be *is* penalized (Eq. (4)).
    penal_X_be, Dpenal_X_be_DX_be = penalize(X_be,
                                             OPT['parameters']['penalization_param'],
                                             OPT['parameters']['penalization_scheme'])

    # Sensitivities of effective (penalized) densities, Eq. (26)
    Dpenal_X_be_Dbar_s = Dpenal_X_be_DX_be[:, :, None] * DX_be_Dbar_s
    Dpenal_X_be_Dbar_size = Dpenal_X_be_DX_be * DX_be_Dbar_size
    Dpenal_X_be_Dbar_radii = Dpenal_X_be_DX_be * DX_be_Dbar_radii

    # Combined density of Eq. (5).
    penal_rho_e, Dpenal_rho_e_Dpenal_X_be = smooth_max(penal_X_be,
                                                       OPT['parameters']['smooth_max_param'],
                                                       OPT['parameters']['smooth_max_scheme'],
                                                       FE['material']['rho_min'])

    # Sensitivities of combined densities, Eq. (25)
    Dpenal_rho_e_Dbar_s = Dpenal_rho_e_Dpenal_X_be[:,
                                                   :, None] * Dpenal_X_be_Dbar_s
    Dpenal_rho_e_Dbar_size = Dpenal_rho_e_Dpenal_X_be * Dpenal_X_be_Dbar_size
    Dpenal_rho_e_Dbar_radii = Dpenal_rho_e_Dpenal_X_be * Dpenal_X_be_Dbar_radii

    # Sensitivities of projected density
    Dpenal_rho_e_Ddv = np.zeros((FE['n_elem'], OPT['n_dv']))

    # Stack together sensitivities with respect to different design
    # variables into a single vector per element
    for b in range(0, GEOM['n_bar']):
        Dpenal_rho_e_Ddv[:, OPT['bar_dv'][:, b]] = \
            Dpenal_rho_e_Ddv[:, OPT['bar_dv'][:, b]] + np.concatenate(
                (Dpenal_rho_e_Dbar_s[b, :, :].reshape((FE['n_elem'], 2*FE['dim']), order='F'),
                 Dpenal_rho_e_Dbar_size[b, :].reshape((FE['n_elem'], 1)),
                 Dpenal_rho_e_Dbar_radii[b, :].reshape((FE['n_elem'], 1))),
                axis=1)

    # Write the element densities and their sensitivities to OPT
    OPT['elem_dens'] = rho_e
    OPT['Delem_dens_Ddv'] = Drho_e_Ddv
    OPT['penalized_elem_dens'] = penal_rho_e
    OPT['Dpenalized_elem_dens_Ddv'] = Dpenal_rho_e_Ddv

    # Remark!!
    '''Reinitializing the simp density at every iteration to overall the bars with matrix material'''
    # OPT['simp']['elem_dens'][penal_rho_e > 1] = 1.0
    # if OPT['simp']['iteration'] < 40:
    #     OPT['simp']['elem_dens'][penal_rho_e > 0.8] = 1.0
    # else:
    #     OPT['simp']['elem_dens'][penal_rho_e > 1] = 1.0

    OPT['simp']['plot_dens'] = OPT['simp']['elem_dens']
    # OPT['simp']['plot_dens'] = OPT['elem_dens']

    compute_bar_material_orientation(FE, GEOM)

    C_void = FE['void']['C']
    # FE['elem_C_proj'] = np.zeros((3, 3, FE['n_elem']))
    FE['elem_C_proj'] = np.zeros((3, 3, FE['n_elem']))
    FE['KE_proj'] = np.zeros((8, 8, n_bar, FE['n_elem']))

    for nbar in range(n_bar):
        FE['elem_C_proj'][:, :, :] = FE['material']['Cb'][:, :, nbar][:, :, None]
        FE['KE_proj'][:, :, nbar, :] = compute_bar_element_stiffness(
            FE, FE['elem_C_proj'])

    FE['Ke_proj'] = np.sum(FE['KE_proj'], axis=2)


def smooth_max(x, p, form_def, x_min):
    #
    # This def computes a smooth approximation of the maximum of x.  The
    # type of smooth approximation (listed below) is given by the argument
    # form_def, and the corresponding approximation parameter is given by p.
    # x_min is a lower bound to the smooth approximation for the modified
    # p-norm and modified p-mean approximations.
    #
    #
    #     The optional third argument is a string that indicates the way the
    #	  approximation is defined, possible values are:
    # 		'mod_p-norm'   : overestimate using modified p-norm (supports x=0)
    # 		'mod_p-mean'   : underestimate using modified p-norm (supports x=0)
    #		'KS'           : Kreisselmeier-Steinhauser, overestimate
    #		'KS_under'     : Kreisselmeier-Steinhauser, underestimate
    #

    if form_def == 'mod_p-norm':
        # Eq. (6)
        # in this case, we assume x >= 0
        S = (x_min**p + (1-x_min**p)*np.sum(x**p, axis=0))**(1/p)
        dSdx = (1-x_min**p)*(x/S)**(p-1)

    elif form_def == 'mod_p-mean':
        # in this case, we assume x >= 0
        N = x.shape[0]
        S = (x_min**p + (1-x_min**p)*np.sum(x**p, axis=0)/N)**(1/p)
        dSdx = (1-x_min**p)*(1/N)*(x/S)**(p-1)

    elif form_def == 'KS':
        epx = np.exp(p*x)
        sum_epx = np.sum(epx, axis=0)

        S = x_min + (1-x_min) * np.log(sum_epx)/p
        dSdx = (1-x_min) * epx / sum_epx
    elif form_def == 'KS_under':
        # note: convergence might be fixed with Euler-Gamma
        N = x.shape[0]
        epx = np.exp(p*x)
        sum_epx = np.sum(epx, axis=0)

        S = x_min + (1-x_min)*np.log(sum_epx/N) / p
        dSdx = (1-x_min) * epx / sum_epx
    else:
        print('\nsmooth_max received invalid form_def.\n')

    return S, dSdx


def update_dv_from_geom(FE, OPT, GEOM):
    #
    # This def updates the values of the design variables (which will be
    # scaled if OPT.options['dv']_scaling is true) based on the unscaled bar
    # geometric parameters. It does the opposite from the def
    # update_geom_from_dv.
    #

    # global GEOM, OPT

    # Fill in design variable vector based on the initial design
    # Eq. (32
    OPT['dv'][OPT['point_dv'], 0] = ((GEOM['initial_design']['point_matrix'][:, 1:] -
                                      OPT['scaling']['point_min']) / OPT['scaling']['point_scale']).flatten()

    OPT['dv'][OPT['size_dv'], 0] = GEOM['initial_design']['bar_matrix'][:, -2].copy()

    OPT['dv'][OPT['radius_dv'], 0] = (GEOM['initial_design']['bar_matrix'][:, -1]
                                      - OPT['scaling']['radius_min']) / OPT['scaling']['radius_scale']


def update_geom_from_dv(FE, OPT, GEOM):
    # This def updates the values of the unscaled bar geometric parameters
    # from the values of the design variableds (which will be scaled if
    # OPT.options['dv']_scaling is true). It does the
    # opposite from the def update_dv_from_geom.
    #
    # global GEOM , OPT , FE

    # Eq. (32)
    GEOM['current_design']['point_matrix'][:, 1:] = (OPT['scaling']['point_scale'][:, None] *
                                                     OPT['dv'][OPT['point_dv']].reshape((FE['dim'], GEOM['n_point']), order='F') +
                                                     OPT['scaling']['point_min'][:, None]).T

    GEOM['current_design']['bar_matrix'][:, -
                                         2] = OPT['dv'][OPT['size_dv']].copy().flatten()

    GEOM['current_design']['bar_matrix'][:, -1] = (OPT['dv'][OPT['radius_dv']] *
                                                   OPT['scaling']['radius_scale'] +
                                                   OPT['scaling']['radius_min']).flatten()

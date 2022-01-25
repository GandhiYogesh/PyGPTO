import numpy as np
import scipy.sparse as sp

from geometry_projection import *


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

    DRR_b_Dn_b[I, I, :, :] = 2*R_b[I, I, :, None] * \
        DR_b_Dx_2b[I, I, :, :]  # (3,b,dim)
    DRR_b_Dn_b[J, I, :, :] = R_b[i, I, :, None]*DR_b_Dx_2b[j,
                                                           I, :, :] + DR_b_Dx_2b[i, I, :, :]*R_b[j, I, :, None]
    DRR_b_Dn_b[I, J, :, :] = 2*R_b[I, i, :, None]*DR_b_Dx_2b[I,
                                                             j, :, :] + 2*DR_b_Dx_2b[I, i, :, :] * R_b[I, j, :, None]
    DRR_b_Dn_b[J, J, :, :] = R_b[i, i, :, None]*DR_b_Dx_2b[j, j, :, :] + DR_b_Dx_2b[i, i, :, :] * R_b[j,
                                                                                                      j, :, None] + R_b[i, j, :, None]*DR_b_Dx_2b[j, i, :, :] + DR_b_Dx_2b[i, j, :, :] * R_b[j, i, :, None]

    if FE['dim'] == 2:
        RR_b = RR_b[[0, 1, 5], :, :][:, [0, 1, 5], :]
        DRR_b_Dn_b = DRR_b_Dn_b[[0, 1, 5], :, :][:, [0, 1, 5], :]

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
        FE['material']['DCb_Dbar_vector'][:, :, nbar,
                                          :] = DC_Dbar_vec/point_scale[None, None, d]


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


def compute_relevent_bar_elem_distance(FE, OPT, GEOM, *args):
    tol = 1e-12
    b_set = args[0]
    e_set = args[1]
    n_elem = len(e_set)
    dim = FE['dim']

    points = GEOM['current_design']['point_matrix'][:, 1:].T  # (2,16)

    x_1b = points.T.flatten()[OPT['bar_dv'][0:dim, b_set]]  # (2,b) (2,8)
    x_2b = points.T.flatten()[OPT['bar_dv'][dim:2*dim, b_set]]

    x_e = FE['centroids'][:, e_set].reshape(dim, n_elem)  # (2,n_elem)

    a_b = x_2b - x_1b
    l_b = np.sqrt(np.sum(a_b**2, 0))
    if l_b < tol:
        l_b = 1

    a_b = np.divide(a_b, l_b).reshape(dim, 1)

    x_e_1b = (x_e.T[:, None] - x_1b.T).swapaxes(0, 2)
    x_e_2b = (x_e.T[:, None] - x_2b.T).swapaxes(0, 2)

    norm_x_e_1b = np.sqrt(np.sum(x_e_1b**2, 0))
    norm_x_e_2b = np.sqrt(np.sum(x_e_2b**2, 0))

    l_be = np.sum(x_e_1b*a_b[:, :, None], 0)
    vec_r_be = x_e_1b - (l_be.T*a_b[:, None]).swapaxes(1, 2)
    r_be = np.sqrt(np.sum(vec_r_be**2, 0))

    l_be_over_l_b = (l_be.T / l_b).T

    branch1 = l_be <= 0.0
    branch2 = l_be_over_l_b >= 1
    branch3 = np.logical_not(np.logical_or(branch1, branch2))

    dist = branch1 * norm_x_e_1b + \
        branch2 * norm_x_e_2b + \
        branch3 * r_be

    Dd_be_Dx_1b = np.zeros((FE['dim'], 1, n_elem))
    Dd_be_Dx_2b = np.zeros((FE['dim'], 1, n_elem))

    d_inv = dist**(-1)
    d_inv[np.isinf(d_inv)] = 0

    Dd_be_Dx_1b = -x_e_1b * d_inv * branch1 + \
        -vec_r_be * d_inv * (1 - l_be_over_l_b) * branch3

    Dd_be_Dx_2b = -x_e_2b * d_inv * branch2 + \
        -vec_r_be * d_inv * l_be_over_l_b * branch3

    Dd_be_Dbar_ends = np.concatenate((Dd_be_Dx_1b, Dd_be_Dx_2b),
                                     axis=0).transpose((1, 2, 0)) * \
        np.concatenate((OPT['scaling']['point_scale'],
                       OPT['scaling']['point_scale']))

    return Dd_be_Dbar_ends


def project_relevent_element_densities_and_stiffness(FE, GEOM, OPT):
    # Desgin Variables
    dim = FE['dim']
    n_bar = GEOM['n_bar']
    n_dv = OPT['n_dv']
    n_bar_dv = OPT['bar_dv'].shape[0]
    bar_ends = np.array(range(0, 2*FE['dim']))
    bar_radii = bar_ends[-1] + 1
    bar_size = bar_radii + 1

    # Number of elements and nodes.
    elements_per_side = FE['mesh_input']['elements_per_side']
    FE['n_elem'] = np.prod(elements_per_side[:])
    FE['n_node'] = np.prod(elements_per_side[:]+1)

    # Distance from x_e to the medial surface
    d_be, Dd_be_Dbar_ends = compute_bar_elem_distance(FE, OPT, GEOM)

    r_b = GEOM['current_design']['bar_matrix'][:, -1]  # bar radii
    # sample window radius
    r_e_full = OPT['parameters']['elem_r']
    X_be_full = (r_b[:, None] - d_be) / r_e_full[None, :]   # signed distance

    FE['bar_element_bool'] = X_be_full > -1

    b_ind, e_ind = np.where(FE['bar_element_bool'])
    n_be = len(b_ind)

    E = np.unique(e_ind)                                # list of relevent elem
    n_E = len(E)
    E_ind = sp.csr_matrix(
        (E, (np.arange(0, len(E)), np.zeros(len(E), dtype=int))))

    rho_eff_be = np.zeros((GEOM['n_bar'], n_E))
    Drho_eff_be_Ddv = np.zeros(
        shape=(GEOM['n_bar'], n_E, OPT['n_dv']), dtype=float)

    penal_rho_eff_be = np.zeros((GEOM['n_bar'], n_E))
    Dpenal_rho_eff_be_Ddv = np.zeros(
        shape=(GEOM['n_bar'], n_E, OPT['n_dv']), dtype=float)

    be_ind_list = [[]]*GEOM['n_bar']
    e_set_list = [[]]*GEOM['n_bar']
    E_set_list = [[]]*GEOM['n_bar']
    for b in range(GEOM['n_bar']):
        be_ind_list[b] = np.array(np.where(b_ind == b)).flatten()
        e_set_list[b] = e_ind[be_ind_list[b]].reshape(be_ind_list[b].size, 1)
        E_set_list[b] = np.searchsorted(
            E_ind.toarray().flatten(), np.intersect1d(E_ind.toarray(), e_set_list[b]))

        be_ind = be_ind_list[b]
        e_set = e_set_list[b]
        E_set = E_set_list[b]

        Dd_be_Dbar_ends = compute_relevent_bar_elem_distance(
            FE, OPT, GEOM, b, e_set)

        X_be = X_be_full[b, e_ind[be_ind]]
        rho_be = np.zeros(X_be.size)
        Drho_be_Dx_be = np.zeros(X_be.size)

        inB = np.abs(X_be) < 1   # in the boundary
        ins = 1 <= X_be          # inside the bar

        rho_be[ins] = 1

        if FE['dim'] == 2:
            rho_be[inB] = 1 + (X_be[inB]*np.sqrt(1.0 - X_be[inB]
                                                 ** 2) - np.arccos(X_be[inB])) / np.pi
            Drho_be_Dx_be[inB] = (np.sqrt(1.0 - X_be[inB]**2)
                                  * 2.0) / np.pi

        if r_e_full.size == 1:
            r_e = r_e_full
        else:
            r_e = r_e_full[e_ind[be_ind]]

        # Sensitivities of raw projected densities, Eqs. (27) and (29)
        Drho_be_Dbar_ends = (Drho_be_Dx_be * -1/r_e *
                             Dd_be_Dbar_ends.transpose((2, 0, 1))).transpose((1, 2, 0))

        Drho_be_Dbar_radii = OPT['scaling']['radius_scale'] * \
            Drho_be_Dx_be * np.transpose(1/r_e)

        penal_rho_be, Dpenal_rho_be_Drho_be = penalize(rho_be,
                                                       OPT['parameters']['penalization_param'],
                                                       OPT['parameters']['penalization_scheme'])
        # Sensitivities of effective (penalized) densities, Eq. (26)
        Dpenal_rho_be_Dbar_radii = Dpenal_rho_be_Drho_be * Drho_be_Dbar_radii
        Dpenal_rho_be_Dbar_ends = Dpenal_rho_be_Drho_be[:,
                                                        None]*Dd_be_Dbar_ends

        alpha_b = GEOM['current_design']['bar_matrix'][b, -2]  # bar size
        penal_alpha_b, Dpenal_alpha_b_Dalpha_b = penalize(alpha_b,
                                                          OPT['parameters']['penalization_param'],
                                                          OPT['parameters']['penalization_scheme'])

        rho_eff_be[b, E_set] = alpha_b * rho_be
        Drho_eff_be_Ddv[b, E_set[:, None], OPT['bar_dv']
                        [bar_ends, b]] = alpha_b*Drho_be_Dbar_ends
        Drho_eff_be_Ddv[b, E_set, OPT['bar_dv']
                        [bar_radii, b]] = alpha_b*Drho_be_Dbar_radii
        Drho_eff_be_Ddv[b, E_set, OPT['bar_dv'][bar_size, b]] = rho_be

        penal_rho_eff_be[b, E_set] = penal_alpha_b * penal_rho_be
        Dpenal_rho_eff_be_Ddv[b, E_set[:, None], OPT['bar_dv']
                              [bar_ends, b]] = penal_alpha_b*Dpenal_rho_be_Dbar_ends
        Dpenal_rho_eff_be_Ddv[b, E_set, OPT['bar_dv']
                              [bar_radii, b]] = penal_alpha_b*Dpenal_rho_be_Dbar_radii
        Dpenal_rho_eff_be_Ddv[b, E_set, OPT['bar_dv']
                              [bar_size, b]] = penal_rho_be

    FE['relevent_element_list'] = E

    compute_bar_material_orientation(FE, GEOM)

    C_void = FE['void']['C']

    # FE['elem_C_proj'] = np.zeros((3, 3, E.size))
    FE['elem_C_proj'] = np.tile(FE['void']['C'][:, :, None], FE['n_elem'])
    DC_bar_Ddv = np.zeros((3, 3, n_bar, n_dv))

    for nbar in range(n_bar):
        E_set = E_set_list[nbar]

        if E_set.size > 0:
            FE['elem_C_proj'][:, :, E_set] = FE['elem_C_proj'][:, :, E_set] + \
                (FE['material']['Cb'][:, :, 0] - C_void)[:, :, None] * \
                penal_rho_eff_be[0, E_set][None, None, :]

        bar_end1 = bar_ends[:dim]
        bar_end2 = bar_ends[dim:2*dim]

        DC_bar_Ddv[:, :, nbar, OPT['bar_dv'][bar_end2, nbar]
                   ] = FE['material']['DCb_Dbar_vector'][:, :, nbar, :]
        DC_bar_Ddv[:, :, nbar, OPT['bar_dv'][bar_end1, nbar]] = - \
            FE['material']['DCb_Dbar_vector'][:, :, nbar, :]

    FE['Delem_C_proj_Ddv'] = np.zeros((3, 3, E.size, n_dv))

    for nbar in range(n_bar):
        E_set = E_set_list[nbar]
        be_ind = be_ind_list[nbar]
        Ib_len = b_ind[be_ind].size
        C_diff_tmp = np.zeros((3, 3, Ib_len))
        for ind, nbar in zip(range(Ib_len), b_ind[be_ind]):
            C_diff_tmp[:, :, ind] = FE['material']['Cb'][:, :, nbar] - C_void

        Drho_mat_tmp = Dpenal_rho_eff_be_Ddv[None, None, nbar, E_set, :]
        rho_mat_tmp = penal_rho_eff_be[None, None, nbar, E_set, None]
        DC_bar_Ddv_tmp = DC_bar_Ddv[:, :, nbar, None, :]

        FE['Delem_C_proj_Ddv'][:, :, E_set, :] = FE['Delem_C_proj_Ddv'][:, :, E_set,
                                                                        :] + Drho_mat_tmp*C_diff_tmp[:, :, :, None] + rho_mat_tmp*DC_bar_Ddv_tmp

    FE['Ke_proj'] = compute_bar_element_stiffness(FE, FE['elem_C_proj'])


def init_orthotropic_lamina(FE, OPT):
    vf = OPT['functions']['constraint_limit'][0]
    # vf = OPT['functions']['f'][1]['value']  # bar volume fraction.
    vm = OPT['simp']['constraint_limit']    # 0.5 Fix value!!

    # lamina material information
    lE1 = FE['material']['E']
    lE2 = FE['simp']['matrix']['E']
    lv12 = FE['material']['nu']*(vf+vm)     # 0.23: Poisson's ratio
    lv21 = (lv12*lE2)/lE1

    lv12 = 0.25
    lv21 = 0.25
    lG1 = lE1/(2*(1+vf))
    lG2 = lE2/(2*(1+vm))
    lG12 = lG1*lG2/(lG1*vm + lG2*vf)        # 0.65 : Shear modulus

    # Testing constant value for orthotropic material distribution.
    lE1 = 25
    lE2 = 1
    lG12 = 0.5

    #    Reduced Stiffness matrix and layer thickness
    h0 = 0.1
    Ql = 1/(1-lv12*lv21)*np.array([[lE1, lv21*lE1, 0],
                                   [lv12*lE2, lE2, 0],
                                   [0, 0, lG12*(1-lv12*lv21)]])

    Ql = np.around(Ql, 2)
    FE['material']['Ql'] = Ql

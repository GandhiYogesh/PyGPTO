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


def compute_bar_material_orientation(FE, GEOM):
    point_mat = GEOM['current_design']['point_matrix']
    bar_mat = GEOM['current_design']['bar_matrix']

    bar_tol = 1e-12  # threshold below which bar is just a circle
    n_bar = bar_mat.shape[0]

    x_1b = np.zeros((3, n_bar))
    x_2b = np.zeros((3, n_bar))  # these are always in 3D

    pt1_IDs = bar_mat[:, 1].astype(int)
    pt2_IDs = bar_mat[:, 2].astype(int)

    x_1b[0:FE['dim'], :] = point_mat[GEOM['point_mat_row'][pt1_IDs].toarray()[
        :, 0], 1:].T
    x_2b[0:FE['dim'], :] = point_mat[GEOM['point_mat_row'][pt2_IDs].toarray()[
        :, 0], 1:].T

    n_b = x_2b - x_1b
    l_b = np.sqrt(np.sum(n_b*n_b, 0))[None, :]

    # principle bar direction
    e_hat_1b = n_b/l_b
    short = l_b < bar_tol
    if short.any():
        e_hat_1b[:, short[0, :]] = np.tile(
            np.array([[1], [0], [0]]), (1, sum(short)))

    # determine coordinate direction most orthogonal to bar
    case_1 = (abs(n_b[0, :]) < abs(n_b[1, :])) & (
        abs(n_b[0, :]) < abs(n_b[2, :]))
    case_2 = (abs(n_b[1, :]) < abs(n_b[0, :])) & (
        abs(n_b[1, :]) < abs(n_b[2, :]))
    case_3 = np.logical_not(case_1 | case_2)

    # secondary bar direction
    e_alpha = np.zeros(n_b.shape)
    e_alpha[0, case_1] = 1
    e_alpha[1, case_2] = 1
    e_alpha[2, case_3] = 1

    e_2b = l_b * np.cross(e_alpha, e_hat_1b, axis=0)
    norm_e_2b = np.sqrt(np.sum(e_2b**2, 0))
    e_hat_2b = e_2b/norm_e_2b

    # tertiary bar direction
    e_3b = np.cross(e_hat_1b, e_hat_2b, axis=0)
    norm_e_3b = np.sqrt(sum(e_3b**2))
    e_hat_3b = e_3b/norm_e_3b

    # Jacobian transformation (rotation) matrix R
    R_b = np.zeros((3, 3, n_bar))
    R_b[:, 0, :] = e_hat_1b
    R_b[:, 1, :] = e_hat_2b
    R_b[:, 2, :] = e_hat_3b

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

    if FE['dim'] == 2:
        RR_b = RR_b[[0, 1, 5], :, :][:, [0, 1, 5], :]

    FE['bar_orientations'] = RR_b

    for i in range(RR_b.shape[2]):
        if(np.linalg.det(RR_b[:, :, i])):
            pass
        else:
            print("Singular Matrix")


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


def project_relevent_element_densities_and_stiffness(FE, GEOM, OPT):
    # Desgin Variables
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

    compute_bar_material_orientation(FE, GEOM)

    FE['material']['Cb'] = np.zeros((3, 3, GEOM['n_bar']))

    for b in range(GEOM['n_bar']):
        C = np.dot(np.dot(FE['bar_orientations'][:, :, b], FE['material']
                   ['Ql']), np.transpose(FE['bar_orientations'][:, :, b]))
        FE['material']['Cb'][:, :, b] = C

    # FE['elem_C_proj'] = np.tile(FE['void']['C'][:, :, None], FE['n_elem'])
    FE['elem_C_proj'] = np.zeros((3, 3, FE['n_elem']))

    for b in range(GEOM['n_bar']):
        E_set = E_set_list[b]
        Cdiff = FE['material']['Cb'][:, :, b] - FE['void']['C']
        FE['elem_C_proj'][:, :, E_set] = (FE['elem_C_proj'][:, :, E_set] + Cdiff[:, :, None]) * \
            penal_rho_eff_be[b, E_set][:, None, None].transpose(1, 2, 0)

    FE['Ke_proj'] = compute_bar_element_stiffness(FE, FE['elem_C_proj'])

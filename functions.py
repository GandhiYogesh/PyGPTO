import numpy as np
from geometry_projection import *
from FE_routines import *
# from LP_routines_v1 import *


def compute_compliance(FE, OPT, GEOM):
    # This function computes the mean compliance and its sensitivities
    # based on the last finite element analysis
    # global FE, OPT

    # compute the compliance (Eq. (15))
    c = np.dot(FE['U'].flatten(), FE['P'].toarray().flatten())

    # compute the design sensitivity
    # Ke = FE['Ke']
    Ke = FE['Ke_proj']

    Ue = FE['U'][FE['edofMat']].repeat(
        FE['n_edof'], axis=2).transpose((1, 2, 0))
    Ue_T = Ue.transpose((1, 0, 2))

    Dc_Dpenalized_elem_dens = np.sum(np.sum(
        - Ue_T * Ke * Ue,
        0), 0)   # Eq. (24)

    # Eq. (25)
    Dc_Ddv = Dc_Dpenalized_elem_dens @ OPT['Dpenalized_elem_dens_Ddv']
    grad_c = Dc_Ddv.T

    # grad_c = np.zeros((OPT['n_dv']))
    # Ii, Ij, rel_elem_ind, dv_ind = np.where(FE['Delem_C_proj_Ddv'] != 0)
    # dv_set = np.unique(dv_ind)

    # for i in range(dv_set.size):
    #     dv = dv_set[i]
    #     E_set = np.unique(rel_elem_ind[dv_ind == dv])
    #     DC_Ddv_tmp = FE['Delem_C_proj_Ddv'][:, :, E_set, dv]

    #     DKE_proj_Ddv = compute_bar_element_stiffness(FE, DC_Ddv_tmp)
    #     edofs = FE['edofMat'][FE['relevent_element_list'][E_set], :].T

    #     Ue = FE['U'][edofs].transpose(0, 2, 1)
    #     Ue_trans = Ue.transpose(1, 0, 2)

    #     Dc_Ddv = -Ue_trans*DKE_proj_Ddv*Ue
    #     grad_c[dv] = np.sum(Dc_Ddv.ravel())

    # save these values in the OPT structure
    OPT['compliance'] = c
    OPT['grad_compliance'] = grad_c

    return c, grad_c


def compute_volume_fraction(FE, OPT, GEOM):
    #
    # This function computes the volume fraction and its sensitivities
    # based on the last geometry projection
    #
    # global FE, OPT

    # compute the volume fraction
    v_e = FE['elem_vol']  # element
    V = np.sum(v_e)  # full volume
    v = np.dot(v_e, OPT['elem_dens'])  # projected volume
    volfrac = v/V  # Eq. (16)

    # compute the design sensitivity
    Dvolfrac_Ddv = (v_e @ OPT['Delem_dens_Ddv'])/V   # Eq. (31)
    grad_volfrac = Dvolfrac_Ddv

    # output
    OPT['volume_fraction'] = volfrac
    OPT['grad_volume_fraction'] = grad_volfrac

    return volfrac, grad_volfrac


def evaluate_relevant_functions(FE, OPT, GEOM):
    # Evaluate_relevant_functions() looks at OPT['functions'] and evaluates the
    # relevant functions for this problem based on the current OPT['dv']
    # global OPT

    OPT['functions']['n_func'] = len(OPT['functions']['f'])

    for i in range(0, OPT['functions']['n_func']):
        value, grad = eval(OPT['functions']['f'][i]
                           ['function'] + "(FE,OPT,GEOM)")
        OPT['functions']['f'][i]['value'] = value
        OPT['functions']['f'][i]['grad'] = grad


def nonlcon(dv):
    global FE, OPT, GEOM

    OPT['dv_old'] = OPT['dv'].copy()
    OPT['dv'] = dv.copy()

    # Update or perform the analysis
    if (OPT['dv'] != OPT['dv_old']).any():
        update_geom_from_dv(FE, OPT, GEOM)
        perform_analysis(FE, OPT, GEOM)

    n_con = OPT['functions']['n_func']-1  # number of constraints
    g = np.zeros((n_con, 1))

    for i in range(0, n_con):
        g[i] = OPT['functions']['f'][i+1]['value']
        g = g - OPT['functions']['constraint_limit']

    return g.flatten()


def nonlcongrad(dv):
    global FE, OPT, GEOM

    n_con = OPT['functions']['n_func']-1  # number of constraints
    gradg = np.zeros((OPT['n_dv'], n_con))

    for i in range(0, n_con):
        gradg[:, i] = OPT['functions']['f'][i+1]['grad']

    return gradg.flatten()


def obj(dv):
    global FE, OPT, GEOM

    OPT['dv_old'] = OPT['dv'].copy()  # save the previous design
    OPT['dv'] = dv.copy()  # update the design

    # If different, update or perform the analysis
    if (OPT['dv'] != OPT['dv_old']).any():
        update_geom_from_dv(FE, OPT, GEOM)
        perform_analysis(FE, OPT, GEOM)

    f = OPT['functions']['f'][0]['value'].flatten().copy()
    g = OPT['functions']['f'][0]['grad'].copy()

    return f, g


def perform_analysis(FE, OPT, GEOM):
    # Perform the geometry projection, solve the finite
    # element problem for the displacements and reaction forces, and then
    # evaluate the relevant functions.
    project_element_densities(FE, OPT, GEOM)
    # project_relevent_element_densities_and_stiffness(FE, GEOM, OPT)
    FE_analysis(FE, OPT, GEOM)
    evaluate_relevant_functions(FE, OPT, GEOM)

import numpy as np
import scipy.sparse as sp

from FE_routines import*


def lk():
    E = FE['simp']['matrix']['E']
    nu = FE['simp']['matrix']['nu']
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu /
                 8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

    return (KE)


def run_simp():
    nelx = FE['mesh_input']['elements_per_side'][0]
    nely = FE['mesh_input']['elements_per_side'][1]
    p = OPT['parameters']['penalization_param']
    vol = OPT['simp']['constraint_limit']

    E1 = FE['simp']['matrix']['E']
    E0 = FE['void']['Emin']
    E2 = FE['material']['E']

    x1vol = vol * np.ones(nely*nelx, dtype=float)
    x1 = OPT['simp']['elem_dens']
    x2 = OPT['elem_dens']

    E = x2**(p)*E2+x1**(p)*E1
    dEdx1 = x1**(p-1)*(x2**(p)*(E2-E1) + E1)
    dEdx2 = x1**(p)*x2**(p-1)*(E2-E1)

    gS = 0  # must be initialized to use the NGuyen/Paulino OC approach
    k0 = lk()  # Element stiffness matrix with unit Young Modulus
    h = OPT['simp']['H']
    hs = OPT['simp']['Hs']
    dv = OPT['simp']['dv']
    dc = OPT['simp']['dce']
    ce = OPT['simp']['ce']

    ce[:] = E*(np.dot(FE['U'][FE['edofMat']].reshape(
        nelx*nely, 8), k0)*FE['U'][FE['edofMat']].reshape(nelx*nely, 8)).sum(1)
    dc[:] = -p*(np.abs(dEdx1 + dEdx2))
    dv[:] = np.ones(nely*nelx)

    # filter
    dc[:] = np.asarray(h*(dc[np.newaxis].T/hs))[:, 0]
    dv[:] = np.asarray(h*(dv[np.newaxis].T/hs))[:, 0]

    # Optimality Criterion
    l1 = 0
    l2 = 1e9
    l = (l2-l1)/(l1+l2)
    move_OC = 0.2
    x1new = np.zeros(nelx*nely)
    while l > 1e-3:
        lmid = 0.5*(l2+l1)
        x1new[:] = np.maximum(0.0, np.maximum(
            x1-move_OC, np.minimum(1.0, np.minimum(x1+move_OC, x1*np.sqrt(-dc/dv/lmid)))))
        gt = gS+np.sum((dv*(x1new-x1vol)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        l = (l2-l1)/(l1+l2)
    x1[:] = np.asarray(h*x1new[np.newaxis].T/hs)[:, 0]
    OPT['simp']['elem_dens'] = x1


def density_and_sensitivity_filter():
    nelx = FE['mesh_input']['elements_per_side'][0]
    nely = FE['mesh_input']['elements_per_side'][1]
    rmin = OPT['simp']['rmin']
    nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i*nely+j
            kk1 = int(np.maximum(i-(np.ceil(rmin)-1), 0))
            kk2 = int(np.minimum(i+np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j-(np.ceil(rmin)-1), 0))
            ll2 = int(np.minimum(j+np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k*nely+l
                    fac = rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc+1
    # Finalize assembly and convert to csc format
    H = sp.csc_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely))
    Hs = H.sum(1)
    OPT['simp']['H'] = H
    OPT['simp']['Hs'] = Hs


def init_simp_dens(FE, OPT, GEOM):
    FE['simp'] = {}
    OPT['simp'] = {}

    nelx = FE['mesh_input']['elements_per_side'][0]
    nely = FE['mesh_input']['elements_per_side'][1]

    FE['simp']['matrix'] = {'E': 1, 'nu': 0.3}

    OPT['simp']['iteration'] = 0
    OPT['simp']['rmin'] = 2.0
    OPT['simp']['constraint_limit'] = 0.5
    OPT['simp']['elem_dens'] = OPT['simp']['constraint_limit'] * \
        np.ones(nely*nelx, dtype=float)
    OPT['simp']['dv'] = np.ones(nely*nelx)
    OPT['simp']['dce'] = np.ones(nely*nelx)
    OPT['simp']['ce'] = np.ones(nely*nelx)

    density_and_sensitivity_filter()

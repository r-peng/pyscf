import numpy as np
from pyscf import lib, cc
from pyscf.cc import td_roccd_utils as utils
import scipy
einsum = lib.einsum

def kernel(eris, t, l, tf, step, RK=4):
    no, _, nv, _ = l.shape
    nmo = no + nv
    N = int((tf+step*0.1)/step)
    C = np.eye(nmo, dtype=complex)

    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    d1, d2 = utils.compute_rdm12(t, l)
    e = utils.compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+eris.mf.energy_nuc())) 

    d1_old = np.block([[d1[0],np.zeros((no,nv))],
                       [np.zeros((nv,no)),d1[1]]])
    E = np.zeros(N+1,dtype=complex) 
#    mu = np.zeros((N+1,3),dtype=complex)  
    for i in range(N+1):
        time = i * step
        dt, dl, X, E[i], F = utils.update_RK(t, l, C, eris, time, step, RK)
#        mu[i,:] = einsum('qp,xpq->x',utils.rotate1(d1,C.T.conj()),eris.mu_) 
        # update 
        t += step * dt
        l += step * dl
        C = np.dot(scipy.linalg.expm(-step*X), C)
        d1 = utils.compute_rdm1(t, l)
        d1_new = np.block([[d1[0],np.zeros((no,nv))],
                           [np.zeros((nv,no)),d1[1]]])
        d1_new = utils.rotate1(d1_new, C.T.conj())
        if RK == 1:
            # Ehrenfest error
            err = np.linalg.norm((d1_new-d1_old)/step-1j*F)
            print('time: {:.4f}, EE(mH): {}, X: {}, err: {}'.format(
                  time, (E[i] - E[0]).real*1e3, np.linalg.norm(X), err))
        else:
            print('time: {:.4f}, EE(mH): {}, X: {}'.format(
                  time, (E[i] - E[0]).real*1e3, np.linalg.norm(X)))
        d1_old = d1_new.copy()
    return d1_new, F, C, X, t, l

class ERIs_mol:
    def __init__(self, mf, z=np.zeros(3), w=0.0, td=0.0):
        self.mf = mf
        self.w = w
        self.td = td
        self.h0_, self.h1_, self.eri_ = utils.mo_ints_mol(mf, z)[:3]
        self.picture = 'S'

        # integrals in rotating basis
        self.h0 = np.array(self.h0_, dtype=complex)
        self.h1 = np.array(self.h1_, dtype=complex)
        self.eri = np.array(self.eri_, dtype=complex)

    def rotate(self, C):
        self.h0 = utils.rotate1(self.h0_, C)
        self.h1 = utils.rotate1(self.h1_, C)
        self.eri = utils.rotate2(self.eri_, C)

    def make_tensors(self, time=None):
        no = self.mf.mol.nelec[0]
        h = self.h0.copy()
        if time is not None:
            h += self.h1 * utils.fac_mol(self.w, self.td, time) 

        self.hoo = h[:no,:no].copy()
        self.hvv = h[no:,no:].copy()
        self.hov = h[:no,no:].copy()
        self.oovv = self.eri[:no,:no,no:,no:].copy()
        self.oooo = self.eri[:no,:no,:no,:no].copy()
        self.vvvv = self.eri[no:,no:,no:,no:].copy()
        self.ovvo = self.eri[:no,no:,no:,:no].copy()
        self.ovov = self.eri[:no,no:,:no,no:].copy()
        self.ovvv = self.eri[:no,no:,no:,no:].copy()
        self.ooov = self.eri[:no,:no,:no,no:].copy()

        self.foo  = self.hoo.copy()
        self.foo += 2.0 * einsum('ikjk->ij',self.oooo)
        self.foo -= einsum('ikkj->ij',self.oooo)
        self.fvv  = self.hvv.copy()
        self.fvv += 2.0 * einsum('kakb->ab',self.ovov)
        self.fvv -= einsum('kabk->ab',self.ovvo)
        h = None

class ERIs_sol:
    def __init__(self, mf, z=np.zeros(3), sigma=1.0, w=0.0, td=0.0):
        self.mf = mf
        self.w = w
        self.sigma = sigma
        self.td = td
        self.picture = 'S'
        self.h0_, self.h1_, self.eri_ = utils.mo_ints_cell(mf, z)[:3]

        # integrals in rotating basis
        self.h0 = np.array(self.h0_, dtype=complex)
        self.h1 = np.array(self.h1_, dtype=complex)
        self.eri = np.array(self.eri_, dtype=complex)

    def rotate(self, C, time=None):
        self.h0 = utils.rotate1(self.h0_, C)
        self.h1 = utils.rotate1(self.h1_, C)
        self.eri = utils.rotate2(self.eri_, C)

    def make_tensors(self, time=None):
        no = self.mf.cell.nelec[0]
        h = self.h0.copy()
        if time is not None:
            h += self.h1 * utils.fac_sol(self.sigma, self.w, self.td, time) 

        self.hoo = h[:no,:no].copy()
        self.hvv = h[no:,no:].copy()
        self.hov = h[:no,no:].copy()
        self.oovv = self.eri[:no,:no,no:,no:].copy()
        self.oooo = self.eri[:no,:no,:no,:no].copy()
        self.vvvv = self.eri[no:,no:,no:,no:].copy()
        self.ovvo = self.eri[:no,no:,no:,:no].copy()
        self.ovov = self.eri[:no,no:,:no,no:].copy()
        self.ovvv = self.eri[:no,no:,no:,no:].copy()
        self.ooov = self.eri[:no,:no,:no,no:].copy()

        self.foo  = self.hoo.copy()
        self.foo += 2.0 * einsum('ikjk->ij',self.oooo)
        self.foo -= einsum('ikkj->ij',self.oooo)
        self.fvv  = self.hvv.copy()
        self.fvv += 2.0 * einsum('kakb->ab',self.ovov)
        self.fvv -= einsum('kabk->ab',self.ovvo)
        h = None

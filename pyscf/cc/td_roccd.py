import numpy as np
from pyscf import lib, cc
from pyscf.cc import td_roccd_utils as utils
import scipy
einsum = lib.einsum

def kernel(eris, t, l, tf, step, RK=4, every=1):
    no, _, nv, _ = l.shape
    nmo = no + nv
    N = int((tf+step*0.1)/step)
    C = np.eye(nmo, dtype=complex)

    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    d1, d2 = utils.compute_rdm12(t, l)
#    e = utils.compute_energy(d1, d2, eris, time=None)
#    print('check initial energy: {}'.format(e.real+eris.mf.energy_nuc())) 

    d1_old = np.block([[d1[0],np.zeros((no,nv))],
                       [np.zeros((nv,no)),d1[1]]])
    E = np.zeros(N+1,dtype=complex) 
#    mu = np.zeros((N+1,3),dtype=complex)  
    rdm1 = []
    for i in range(N+1):
        time = i * step
        dt, dl, X, E[i], F = utils.update_RK(t, l, C, eris, time, step, RK)
#        mu[i,:] = einsum('qp,xpq->x',utils.rotate1(d1,C.T.conj()),eris.mu_) 
        # update 
        t += step * dt
        l += step * dl
        C = np.dot(scipy.linalg.expm(-step*X), C)
        if i % every == 0: 
            d1 = utils.compute_rdm1(t, l)
            d1_new = np.block([[d1[0],np.zeros((no,nv))],
                               [np.zeros((nv,no)),d1[1]]])
            d1_new = utils.rotate1(d1_new, C.T.conj())
            rdm1.append(d1_new.copy())
        if RK == 1 and every == 1:
            # Ehrenfest error
            err = np.linalg.norm((d1_new-d1_old)/step-1j*F)
            print('time: {:.4f}, EE(mH): {}, X: {}, err: {}'.format(
                  time, (E[i] - E[0]).real*1e3, np.linalg.norm(X), err))
            d1_old = d1_new.copy()
        else:
            print('time: {:.4f}, EE(mH): {}, X: {}'.format(
                  time, (E[i] - E[0]).real*1e3, np.linalg.norm(X)))
    rdm1 = np.array(rdm1, dtype=complex)
    return rdm1, E
#    return d1_new, F, C, X, t, l

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

    def rotate(self, C):
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

class ERIs_SIAM:
    def __init__(self, model, P=None, mo_energy=None, mo_coeff=None, picture='S'):
        self.model = model
        self.P = P
        self.picture = picture
        self.quick = True
        self.L = model.ll + model.lr + 1

        h  = model.get_tmatS()
        h += model.get_vmatS()
        V = model.get_umatS()
        if mo_coeff is None: 
            F = h.copy()
            F += einsum('pqrs,qs->pr',V ,P)
            mo_energy, mo_coeff = np.linalg.eigh(F)
        self.mo_energy = mo_energy 
        self.mo_coeff  = mo_coeff # t=0 mo_coeff

        # integrals in fixed Bogliubov basis
        self.h_ = utils.rotate1(h, mo_coeff.T)
        self.eri_ = utils.rotate2(V , mo_coeff.T)

        # integrals in rotating basis
        self.h = np.array(self.h_,dtype=complex)
        self.eri = np.array(self.eri_,dtype=complex)

#        if picture == 'I':
#            self.Roo = utils.make_Roo(self.mo_energy, fd)
#            self.Rvv = utils.make_Roo(self.mo_energy, fd_)

    def rotate(self, C):
        self.h = utils.rotate1(self.h_, C)
        self.eri = utils.rotate2(self.eri_, C)

    def make_tensors(self, time=None):
        no = int(self.L/2)
        self.hoo = self.h[:no,:no].copy()
        self.hvv = self.h[no:,no:].copy()
        self.hov = self.h[:no,no:].copy()
        self.oovv = self.eri[:no,:no,no:,no:].copy()
        self.oooo = self.eri[:no,:no,:no,:no].copy()
        self.vvvv = self.eri[no:,no:,no:,no:].copy()
        self.ovvo = self.eri[:no,no:,no:,:no].copy()
        self.ovov = self.eri[:no,no:,:no,no:].copy()
        self.vovo = self.eri[no:,:no,no:,:no].copy()
        self.ovvv = self.eri[:no,no:,no:,no:].copy()
        self.vovv = self.eri[no:,:no,no:,no:].copy()
        self.ooov = self.eri[:no,:no,:no,no:].copy()
        self.oovo = self.eri[:no,:no,no:,:no].copy()
        self.foo, self.fvv = self.hoo.copy(), self.hvv.copy()
        self.foo += einsum('pIqI->pq',self.oooo)
        self.fvv += einsum('pIqI->pq',self.vovo)

#        if self.picture == 'I':
#            self.foo += self.Roo
#            self.fvv += self.Rvv


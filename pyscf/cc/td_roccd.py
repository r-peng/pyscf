import numpy as np
from pyscf import lib, cc
from pyscf.cc import td_roccd_utils as utils
import scipy
einsum = lib.einsum

def build1(d1):
    doo, dvv = d1
    no, nv = doo.shape[0], dvv.shape[0]
    return np.block([[doo,np.zeros((no,nv))],
                     [np.zeros((nv,no)),dvv]])

def kernel(mf, t, l, w, f0, td, tf, step):
    no, _, nv, _ = l.shape
    nmo = no + nv
    N = int((tf+step*0.1)/step)
    C = np.eye(nmo, dtype=complex)

    eris = ERIs(mf, w, f0, td) # in HF basis
    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    d1, d2 = utils.compute_rdm12(t, l)
    e = utils.compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+mf.energy_nuc())) 

    d1_old = build1(d1)
    E = np.zeros(N+1,dtype=complex) 
#    mu = np.zeros((N+1,3),dtype=complex)  
    for i in range(N+1):
        time = i * step
        eris.rotate(C)
        d1, d2 = utils.compute_rdm12(t, l)
        X = utils.compute_X(d1, d2, eris, time)
        dt, dl = utils.update_amps(t, l, eris, time)
        E[i] = utils.compute_energy(d1, d2, eris, time=None) # <H_U>
        F = utils.compute_comm(d1, d2, eris, time) # F_{qp} = <[H_U,p+q]>
#        mu[i,:] = einsum('qp,xpq->x',utils.rotate1(d1,C.T.conj()),eris.mu_) 
        # update 
        t += step * dt
        l += step * dl
        C = np.dot(scipy.linalg.expm(-step*X), C)
        # Ehrenfest error
        d1 = utils.compute_rdm1(t, l)
        d1_new = build1(d1) 
        d1_new = utils.rotate1(d1_new, C.T.conj())
        F = np.block([[F[0],F[1]],[F[2],F[3]]])
        F -= F.T.conj()
        F = utils.rotate1(F, C.T.conj())
        err = np.linalg.norm((d1_new-d1_old)/step-1j*F)
        d1_old = d1_new.copy()
        print('time: {:.4f}, EE(mH): {}, X: {}, err: {}'.format(
              time, (E[i] - E[0]).real*1e3, np.linalg.norm(X), err))
    return d1_new, 1j*F, C, X, t, l

class ERIs:
    def __init__(self, mf, w=0.0, f0=np.zeros(3), td=0.0):
        self.no = mf.mol.nelec[0]
        self.w = w
        self.f0 = f0
        self.td = td
        # integrals in AO basis
        hao = mf.get_hcore()
        eri_ao = mf.mol.intor('int2e_sph')
        mu_ao = mf.mol.intor('int1e_r')
        h1ao = einsum('xuv,x->uv',mu_ao,f0)
        charges = mf.mol.atom_charges()
        coords  = mf.mol.atom_coords()
        self.nucl_dip = einsum('i,ix->x', charges, coords)

        # integrals in HF basis
        mo_coeff = mf.mo_coeff.copy()
        self.h0_ = einsum('uv,up,vq->pq',hao,mo_coeff,mo_coeff)
        self.h1_ = einsum('uv,up,vq->pq',h1ao,mo_coeff,mo_coeff)
        self.mu_ = einsum('xuv,up,vq->xpq',mu_ao,mo_coeff,mo_coeff)
        self.eri_ = einsum('uvxy,up,vr->prxy',eri_ao,mo_coeff,mo_coeff)
        self.eri_ = einsum('prxy,xq,ys->prqs',self.eri_,mo_coeff,mo_coeff)
        self.eri_ = self.eri_.transpose(0,2,1,3)

        # integrals in rotating basis
        self.h0 = np.array(self.h0_, dtype=complex)
        self.h1 = np.array(self.h1_, dtype=complex)
        self.eri = np.array(self.eri_, dtype=complex)

        hao = mu_ao = h1ao = eri_ao = None

    def rotate(self, C, time=None):
        self.h0 = utils.rotate1(self.h0_, C)
        self.h1 = utils.rotate1(self.h1_, C)
        self.eri = utils.rotate2(self.eri_, C)

    def make_tensors(self, time=None):
        no = self.no
        h = utils.full_h(self.h0, self.h1, self.w, self.td, time) 

        self.hoo = h[:no,:no].copy()
        self.hvv = h[no:,no:].copy()
        self.hov = h[:no,no:].copy()
        self.oovv = self.eri[:no,:no,no:,no:].copy()
        self.oooo = self.eri[:no,:no,:no,:no].copy()
        self.vvvv = self.eri[no:,no:,no:,no:].copy()
        self.ovvo = self.eri[:no,no:,no:,:no].copy()
        self.ovov = self.eri[:no,no:,:no,no:].copy()
        self.ovvv = self.eri[:no,no:,no:,no:].copy()
        self.vovv = self.eri[no:,:no,no:,no:].copy()
        self.oovo = self.eri[:no,:no,no:,:no].copy()
        self.ooov = self.eri[:no,:no,:no,no:].copy()

        self.foo  = self.hoo.copy()
        self.foo += 2.0 * einsum('ikjk->ij',self.oooo)
        self.foo -= einsum('ikkj->ij',self.oooo)
        self.fvv  = self.hvv.copy()
        self.fvv += 2.0 * einsum('kakb->ab',self.ovov)
        self.fvv -= einsum('kabk->ab',self.ovvo)

        h = None

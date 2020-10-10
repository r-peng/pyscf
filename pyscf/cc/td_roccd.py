import numpy as np
from pyscf import lib, cc
from pyscf.cc import td_roccd_utils as utils
import scipy
einsum = lib.einsum

#def trace_err(d1, d2, no):
#    # err1 = d_{pp} - N
#    # err2 = d_{prqr}/(N-1) - d_{pq}
#    err1 = abs(np.trace(d1)-no)
#    d2_ = einsum('prqr->pq',d2)
#    d2_ /= no - 1
#    err2 = np.linalg.norm(d2_-d1)
#    d2_ = None
#    return err1, err2

def compute_X(d1, d2, eris, time=None):
    Aovvo = utils.compute_Aovvo(d1)
    _, fov, fvo, _ = utils.compute_comm(d1, d2, eris, time, full=False) 
    no, nv = fov.shape
    Bov = fvo.T - fov.conj()
    Bov = Bov.reshape(no*nv)
    Aovvo = Aovvo.reshape(no*nv,no*nv)
    Xvo = np.dot(np.linalg.inv(Aovvo),Bov)
    Xvo = Xvo.reshape(nv,no)
    return 1j*np.block([[np.zeros((no,no)),Xvo.T.conj()],
                        [Xvo,np.zeros((nv,nv))]])

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

    d1_old = np.block([[d1[0],np.zeros((no,nv))],
                       [np.zeros((nv,no)),d1[1]]])
    E = np.zeros(N+1,dtype=complex) 
#    mu = np.zeros((N+1,3),dtype=complex)  
    for i in range(N+1):
        time = i * step
        eris.rotate(C)
        d1, d2 = utils.compute_rdm12(t, l)
        X = compute_X(d1, d2, eris, time)
        dt, dl = utils.update_amps(t, l, eris, time)
        E[i] = utils.compute_energy(d1, d2, eris, time=None) # <U^{-1}H0U>
#        mu[i,:] = einsum('qp,xpq->x',utils.rotate1(d1,C.T.conj()),eris.mu_) 
        # update 
        t += step * dt
        l += step * dl
        C = np.dot(scipy.linalg.expm(-step*X), C)
        # Ehrenfest error
        d1_new, d2_new = utils.compute_rdm12(t, l)
        B = utils.compute_comm(d1_new, d2_new, eris, time) # F_{qp} = <[U^{-1}HU,p+q]>
        d1_new = np.block([[d1_new[0],np.zeros((no,nv))],
                           [np.zeros((nv,no)),d1_new[1]]])
        d1_new = utils.rotate1(d1_new, C.T.conj())
        B = np.block([[B[0],B[1]],[B[2],B[3]]])
        B -= B.T.conj()
        B = utils.rotate1(B, C.T.conj())
        err = np.linalg.norm((d1_new-d1_old)/step-1j*B)
        d1_old = d1_new.copy()
        print('time: {:.4f}, EE(mH): {}, X: {}, err: {}'.format(
              time, (E[i] - E[0]).real*1e3, np.linalg.norm(X)**2, err))
#    print('trace error: ',tr)
#    print('energy conservation error: ', ec)
    print('imaginary part of energy: ', np.linalg.norm(E.imag))
#    return (E - E[0]).real, (mu - eris.nucl_dip).real
    return d1_new, 1j*B, C, X, t, l

class ERIs:
    def __init__(self, mf, w=0.0, f0=np.zeros(3), td=0.0):
        self.no = mf.mol.nelec[0]
        self.w = w
        self.f0 = f0
        self.td = td
        # integralsin HF basis
        hao = mf.get_hcore()
        eri_ao = mf.mol.intor('int2e_sph')
        mu_ao = mf.mol.intor('int1e_r')
        h1ao = einsum('xuv,x->uv',mu_ao,f0)
        charges = mf.mol.atom_charges()
        coords  = mf.mol.atom_coords()
        self.nucl_dip = einsum('i,ix->x', charges, coords)

        mo_coeff = mf.mo_coeff.copy()
        self.h0_ = einsum('uv,up,vq->pq',hao,mo_coeff,mo_coeff)
        self.h1_ = einsum('uv,up,vq->pq',h1ao,mo_coeff,mo_coeff)
        self.mu_ = einsum('xuv,up,vq->xpq',mu_ao,mo_coeff,mo_coeff)
        self.eri_ = einsum('uvxy,up,vr->prxy',eri_ao,mo_coeff,mo_coeff)
        self.eri_ = einsum('prxy,xq,ys->prqs',self.eri_,mo_coeff,mo_coeff)
        self.eri_ = self.eri_.transpose(0,2,1,3)

        self.h0 = self.h0_.copy()
        self.h1 = self.h1_.copy()
        self.eri = self.eri_.copy()

        muao = h1ao = eri_ao = None

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

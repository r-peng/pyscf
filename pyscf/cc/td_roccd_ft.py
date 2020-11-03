import numpy as np
from pyscf import lib, cc
from pyscf.cc import td_roccd_utils as utils
import scipy
einsum = lib.einsum

def compute_sqrt_fd(mo_energy, beta, mu):
    fd = (mo_energy - mu) * beta
    fd = np.exp(fd) + 1.0
    fd = np.reciprocal(fd) # n_p
    fd_ = 1.0 - fd # 1-n_p
    return np.sqrt(fd), np.sqrt(fd_)

def compute_phys1(d1, eris): 
    # physical rdm1 and commutator 
    # d1 in fixed Bogoliubov basis
    fd, fd_ = eris.fd
    no = eris.no
    d1_  = einsum('pq,p,q->pq',d1[:no,:no],fd ,fd ) 
    d1_ += einsum('pq,p,q->pq',d1[:no,no:],fd ,fd_) 
    d1_ += einsum('pq,p,q->pq',d1[no:,:no],fd_,fd ) 
    d1_ += einsum('pq,p,q->pq',d1[no:,no:],fd_,fd_) 
    return d1_

def kernel(mf, t, l, w, f0, td, tf, beta, mu, step, basis='b'):
    no, _, nv, _ = l.shape
    nmo = no + nv
    N = int((tf+step*0.1)/step)
    if basis == 'b': # for Bogoliubov rotation
        C = np.eye(nmo, dtype=complex)
        eris = ERIs_b(mf, w, f0, td, beta, mu)
        compute_X = utils.compute_X
    else: # for physical rotation
        C = np.eye(no, dtype=complex)
        eris = ERIs_p(mf, w, f0, td, beta, mu)
        compute_X = compute_X_

    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    d1, d2 = utils.compute_rdm12(t, l)
    e = utils.compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+mf.energy_nuc())) 

    d1_old = np.block([[d1[0],np.zeros((no,nv))],
                       [np.zeros((nv,no)),d1[1]]])
    d1_old = compute_phys1(d1_old, eris)
    E = np.zeros(N+1,dtype=complex) 
#    mu = np.zeros((N+1,3),dtype=complex)  
    for i in range(N+1):
        time = i * step
        eris.rotate(C)
        d1, d2 = utils.compute_rdm12(t, l)
        X, _ = compute_X(d1, d2, eris, time)
        dt, dl = utils.update_amps(t, l, eris, time)
        E[i] = utils.compute_energy(d1, d2, eris, time=None) # <H_U>
        F = utils.compute_comm(d1, d2, eris, time) # F_{qp} = <[H_U,p+q]>
        F = np.block([[F[0],F[1]],[F[2],F[3]]])
        F -= F.T.conj()
#        mu[i,:] = einsum('qp,xpq->x',utils.rotate1(d1,C.T.conj()),eris.mu_) 
        # update 
        t += step * dt
        l += step * dl
        C = np.dot(scipy.linalg.expm(-step*X), C)
        # Ehrenfest error
        d1 = utils.compute_rdm1(t, l)
        d1_new = np.block([[d1[0],np.zeros((no,nv))],
                           [np.zeros((nv,no)),d1[1]]])
        if basis == 'b':
            d1_new = utils.rotate1(d1_new, C.T.conj())
            d1_new = compute_phys1(d1_new, eris)
            F = utils.rotate1(F, C.T.conj())
            F = compute_phys1(F, eris)
        else:
            d1_new = compute_phys1(d1_new, eris)
            d1_new = utils.rotate1(d1_new, C.T.conj())
            F = compute_phys1(F, eris)
            F = utils.rotate1(F, C.T.conj())
        err = np.linalg.norm((d1_new-d1_old)/step-1j*F)
        d1_old = d1_new.copy()
        print('time: {:.4f}, EE(mH): {}, X: {}, err: {}'.format(
              time, (E[i] - E[0]).real*1e3, np.linalg.norm(X), err))
    return d1_new, 1j*F, C, X, t, l

def compute_X_(d1, d2, eris, time=None, thresh=1e-10):
    fd, fd_ = eris.fd
    nmo = len(fd)
    _, fov, fvo, _ = utils.compute_comm(d1, d2, eris, time, full=False)  
    F  = einsum('pq,p,q->pq',fov,fd ,fd_)
    F += einsum('pq,p,q->pq',fvo,fd_,fd )
    F -= F.T.conj()
    d1 = np.block([[d1[0],np.zeros((nmo,nmo))],
                   [np.zeros((nmo,nmo)),d1[1]]])
    d1 = compute_phys1(d1, eris)
    A  = einsum('qr,sp->pqrs',np.eye(nmo),d1)
    A -= einsum('sp,qr->pqrs',np.eye(nmo),d1)

    A_, F_ = make_AB(A, F.T)
    R_ = solve(A_, F_, thresh)
    R = make_R(R_, nmo)

    Roo = einsum('pq,p,q->pq',R,fd ,fd )
    Rvv = einsum('pq,p,q->pq',R,fd_,fd_)
    print('check orbital equation: ', np.linalg.norm(F.T-einsum('pqrs,rs->pq',A,R)))
    A = B = Aovvo = Bov = A_ = B_ = R_ = None
    return 1j*R, None 

def extract1(A): # extract A_{p<q,r<s}
    nmo = A.shape[0]
    N = int(nmo*(nmo-1)/2) 
    A_ = np.zeros((N,N))
    row, col = 0,0
    for p in range(nmo):
        for q in range(p+1,nmo):
            for r in range(nmo):
                for s in range(r+1,nmo):
                    A_[row,col] = A[p,q,r,s].copy()
                    col += 1
            row += 1
            col = 0
    return A_

def extract2(A): # extract A_{p<q,rr}
    nmo = A.shape[0]
    N = int(nmo*(nmo-1)/2) 
    A_ = np.zeros((N,nmo))
    row = 0
    for p in range(nmo):
        for q in range(p+1,nmo):
            for r in range(nmo):
                A_[row,r] = A[p,q,r,r].copy()
            row += 1
    return A_

def extract3(A): # extractA_{pp,rr}
    nmo = A.shape[0]
    A_ = np.zeros((nmo,nmo))
    for p in range(nmo):
        for q in range(nmo):
            A_[p,q] = A[p,p,q,q].copy()
    return A_

def make_AB(A, B):
    nmo = B.shape[0] 
    N = int(nmo*(nmo-1)/2)
    iu = np.triu_indices(nmo,1)
    di = np.diag_indices(nmo)

    A11 = extract1( A.real-A.real.transpose(1,0,2,3))
    A12 = extract2( A.real)
    A13 = extract1(-A.imag+A.imag.transpose(1,0,2,3))
    A21 = extract1( A.imag+A.imag.transpose(1,0,2,3))
    A22 = extract2( A.imag)
    A23 = extract1( A.real+A.real.transpose(1,0,2,3))
    A32 = extract3( A.imag)
    A31 = -2 * A22.T 
    A33 = -2 * A12.T
    A_ = np.block([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])

    B_ = np.zeros(2*N+nmo) 
    B_[:N] = B.real[iu].copy() 
    B_[N:2*N] = B.imag[iu].copy()
    B_[2*N:] = B.imag[di].copy()

    A11 = A12 = A13 = A21 = A22 = A23 = A31 = A32 = A33 = None
    return A_, B_

def solve(A, B, thresh=1e-3):
    u, s, vh = np.linalg.svd(A)
    uB = np.dot(u.T.conj(),B)
    idxs = np.argwhere(s>thresh)
    R = np.zeros_like(B)
    for idx in idxs:
        R[idx] = uB[idx]/s[idx]
    R = np.dot(vh.T.conj(), R)
#    print('singularity error: ', np.linalg.norm(B-np.dot(A,R)))
#    print('singular values: ', len(B)-len(idxs))
    return R

def make_R(R_, nmo):
    R = np.zeros((nmo,nmo),dtype=complex)
    N = int(nmo*(nmo-1)/2)
    iu = np.triu_indices(nmo,1)
    di = np.diag_indices(nmo)
    R.real[iu] = R_[:N].copy()
    R.real += R.real.T
    R.real[di] = R_[N:N+nmo].copy()
    R.imag[iu] = R_[N+nmo:].copy()
    R.imag -= R.imag.T
    return R

class ERIs_b:
    def __init__(self, mf, w=0.0, f0=np.zeros(3), td=0.0, beta=0.0, mu=0.0):
        self.no = mf.mol.nao_nr()
        self.w = w
        self.f0 = f0
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.fd = compute_sqrt_fd(mf.mo_energy, beta, mu)
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
        h0 = einsum('uv,up,vq->pq',hao,mo_coeff,mo_coeff)
        h1 = einsum('uv,up,vq->pq',h1ao,mo_coeff,mo_coeff)
        mu = einsum('xuv,up,vq->xpq',mu_ao,mo_coeff,mo_coeff)
        eri = einsum('uvxy,up,vr->prxy',eri_ao,mo_coeff,mo_coeff)
        eri = einsum('prxy,xq,ys->prqs',eri,mo_coeff,mo_coeff)
        eri = eri.transpose(0,2,1,3)
        # integrals in fixed Bogliubov basis
        fd, fd_ = self.fd
        no = self.no
        self.h0_ = np.zeros((no*2,)*2)
        self.h1_ = np.zeros((no*2,)*2)
        self.eri_ = np.zeros((no*2,)*4)
        self.h0_[:no,:no] = einsum('pq,p,q->pq',h0,fd ,fd )
        self.h0_[no:,no:] = einsum('pq,p,q->pq',h0,fd_,fd_)
        self.h0_[:no,no:] = einsum('pq,p,q->pq',h0,fd ,fd_)
        self.h0_[no:,:no] = einsum('pq,p,q->pq',h0,fd_,fd)
        self.h1_[:no,:no] = einsum('pq,p,q->pq',h1,fd ,fd )
        self.h1_[no:,no:] = einsum('pq,p,q->pq',h1,fd_,fd_)
        self.h1_[:no,no:] = einsum('pq,p,q->pq',h1,fd ,fd_)
        self.h1_[no:,:no] = einsum('pq,p,q->pq',h1,fd_,fd )
        self.eri_[:no,:no,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd ,fd ,fd )
        self.eri_[no:,:no,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd ,fd ,fd )
        self.eri_[:no,no:,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd_,fd ,fd )
        self.eri_[:no,:no,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd ,fd_,fd )
        self.eri_[:no,:no,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd ,fd ,fd_)
        self.eri_[:no,:no,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd ,fd_,fd_)
        self.eri_[no:,no:,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd_,fd ,fd )
        self.eri_[:no,no:,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd_,fd_,fd )
        self.eri_[:no,no:,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd_,fd ,fd_)
        self.eri_[no:,:no,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd ,fd ,fd_)
        self.eri_[no:,:no,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd ,fd_,fd )
        self.eri_[:no,no:,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd ,fd_,fd_,fd_)
        self.eri_[no:,:no,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd ,fd_,fd_)
        self.eri_[no:,no:,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd_,fd ,fd_)
        self.eri_[no:,no:,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd_,fd_,fd )
        self.eri_[no:,no:,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fd_,fd_,fd_,fd_)

        # integrals in rotating basis
        self.h0 = np.array(self.h0_, dtype=complex)
        self.h1 = np.array(self.h1_, dtype=complex)
        self.eri = np.array(self.eri_, dtype=complex)

        hao = mu_ao = h1ao = eri_ao = None
        h0 = h1 = mu = eri = None
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

class ERIs_p:
    def __init__(self, mf, w=0.0, f0=np.zeros(3), td=0.0, beta=0.0, mu=0.0):
        self.no = mf.mol.nao_nr()
        self.w = w
        self.f0 = f0
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.fd = compute_sqrt_fd(mf.mo_energy, beta, mu)
        # ZT integrals in HF basis
        hao = mf.get_hcore()
        eri_ao = mf.mol.intor('int2e_sph')
        mu_ao = mf.mol.intor('int1e_r')
        h1ao = einsum('xuv,x->uv',mu_ao,f0)
        charges = mf.mol.atom_charges()
        coords  = mf.mol.atom_coords()
        self.nucl_dip = einsum('i,ix->x', charges, coords)

        mo_coeff = mf.mo_coeff
        self.h0_HF = einsum('uv,up,vq->pq',hao,mo_coeff,mo_coeff)
        self.h1_HF = einsum('uv,up,vq->pq',h1ao,mo_coeff,mo_coeff)
        self.mu_HF = einsum('xuv,up,vq->xpq',mu_ao,mo_coeff,mo_coeff) # dipole
        self.eri_HF = einsum('uvxy,up,vr->prxy',eri_ao,mo_coeff,mo_coeff)
        self.eri_HF = einsum('prxy,xq,ys->prqs',self.eri_HF,mo_coeff,mo_coeff)
        self.eri_HF = self.eri_HF.transpose(0,2,1,3)
        self.mo_energy = mf.mo_energy
        # ZT integrals in rotating basis
        self.h0 = np.array(self.h0_HF, dtype=complex)
        self.h1 = np.array(self.h1_HF, dtype=complex)
        self.eri = np.array(self.eri_HF, dtype=complex)

        hao = eri_ao = muao = h1ao = None

    def rotate(self, C): # rotate ZT integrals
        self.h0 = utils.rotate1(self.h0_HF, C)
        self.h1 = utils.rotate1(self.h1_HF, C)
        self.eri = utils.rotate2(self.eri_HF, C)

    def make_tensors(self, time=None):
        self.h = utils.full_h(self.h0, self.h1, self.w, self.td, time) 
        fd, fd_ = self.fd

        self.hoo = einsum('pq,p,q->pq',self.h,fd ,fd )
        self.hvv = einsum('pq,p,q->pq',self.h,fd_,fd_)
        self.hov = einsum('pq,p,q->pq',self.h,fd ,fd_)
        self.oovv = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd ,fd ,fd_,fd_)
        self.oooo = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd ,fd ,fd ,fd )
        self.vvvv = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd_,fd_,fd_,fd_)
        self.ovvo = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd ,fd_,fd_,fd )
        self.ovov = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd ,fd_,fd ,fd_)
        self.ovvv = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd ,fd_,fd_,fd_)
        self.vovv = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd_,fd ,fd_,fd_)
        self.oovo = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd ,fd ,fd_,fd )
        self.ooov = einsum('pqrs,p,q,r,s->pqrs',self.eri,fd ,fd ,fd ,fd_)

        # "normal orders" the FT tensors
        self.foo  = self.hoo.copy()
        self.foo += 2.0 * einsum('ikjk->ij',self.oooo)
        self.foo -= einsum('ikkj->ij',self.oooo)
        self.fvv  = self.hvv.copy()
        self.fvv += 2.0 * einsum('kakb->ab',self.ovov)
        self.fvv -= einsum('kabk->ab',self.ovvo)

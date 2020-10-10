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

def compute_sqrt_fd(mo_energy, beta, mu):
    fd = (mo_energy - mu) * beta
    fd = np.exp(fd) + 1.0
    fd = np.reciprocal(fd) # n_p
    fd_ = 1.0 - fd # 1-n_p
    return np.sqrt(fd), np.sqrt(fd_)

def solve(A, B, thresh=1e-3):
    u, s, vh = scipy.linalg.svd(A)
    uB = np.dot(u.T.conj(),B)
    idxs = np.argwhere(s>thresh)
    R = np.zeros_like(B)
    for idx in idxs:
        R[idx] = uB[idx]/s[idx]
    R = np.dot(vh.T.conj(), R)
    print('singularity error: ', np.linalg.norm(B-np.dot(A,R)))
    return R

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

def compute_X(d1, d2, eris, time=None, thresh=1e-3):
    fd, fd_ = eris.fd
    nmo = len(fd)
    N = int(nmo*(nmo-1)/2)
    iu = np.triu_indices(nmo,1)
    di = np.diag_indices(nmo)

    Aovvo = utils.compute_Aovvo(d1)
    Aovvo = einsum('pquv,p,q,u,v->pquv',Aovvo,fd,fd_,fd_,fd)
    A = Aovvo - Aovvo.transpose(1,0,3,2).conj() 
    _, fov, fvo, _ = utils.compute_comm(d1, d2, eris, time, full=False)
    Bov = fvo.T - fov.conj()
    Bov = einsum('pq,p,q->pq', Bov, fd, fd_)
    B = Bov - Bov.T.conj() 

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

    tmp = solve(A_, B_, thresh)
    R = np.zeros((nmo,nmo),dtype=complex)
    R.real[iu] = tmp[:N].copy()
    R.real += R.real.T
    R.real[di] = tmp[N:N+nmo].copy()
    R.imag[iu] = tmp[N+nmo:].copy()
    R.imag -= R.imag.T
    # nonzero Xoo, Xvv
    Rvv = einsum('pq,p,q->pq',R,fd_,fd_) 
    Roo = einsum('pq,p,q->pq',R,fd ,fd ) 

    print('check orbital equation: ', np.linalg.norm(B-einsum('pqrs,rs->pq',A,R)))
    Aovvo = fov = fvo = Bov = None
    A = B = A_ = B_ = tmp = None
    A11 = A12 = A13 = A21 = A22 = A23 = A31 = A32 = A33 = None
    return 1j*R, (1j*Roo, 1j*Rvv)  

#def compute_X(d1, d2, eris, time=None, thresh=1e-6):
#    fd, fd_ = eris.fd
#    nmo = len(fd)
#    N = nmo**2
#    Aovvo = utils.compute_Aovvo(d1)
#    Aovvo = einsum('pquv,u,v->pquv',Aovvo,fd_,fd)
#    _, fov, fvo, _ = utils.compute_comm(d1, d2, eris, time, full=False)
#    Bov = fvo.T - fov.conj()
#    A11 =   Aovvo.real + Aovvo.real.transpose(0,1,3,2)
#    A12 = - Aovvo.imag + Aovvo.imag.transpose(0,1,3,2)
#    A21 =   Aovvo.imag + Aovvo.imag.transpose(0,1,3,2)
#    A22 =   Aovvo.real - Aovvo.real.transpose(0,1,3,2)
#    A11 = A11.reshape(N,N)
#    A12 = A12.reshape(N,N)
#    A21 = A21.reshape(N,N)
#    A22 = A22.reshape(N,N)
#    A = np.block([[A11, A12],[A21,A22]])
#    B = np.zeros(N*2)
#    B[:N] = Bov.real.reshape(N)
#    B[N:] = Bov.imag.reshape(N)
#    tmp = solve(A, B) 
#    R = np.zeros((nmo,nmo),dtype=complex)
#    R.real = tmp[:N].reshape(nmo,nmo).copy()
#    R.imag = tmp[N:].reshape(nmo,nmo).copy()
#    R += R.T.conj()
#    print('check orbital equation: ', 
#          np.linalg.norm(Bov-einsum('pqrs,rs->pq',Aovvo,R)))
#    
#    Rvv = einsum('pq,p,q->pq',R,fd_,fd_) 
#    Roo = einsum('pq,p,q->pq',R,fd ,fd ) 
#    Aovvo = Bov = fov = fvo = None
#    A11 = A12 = A21 = A22 = None
#    A = B = tmp = None
#    return 1j*R, (1j*Roo, 1j*Rvv)

def compute_comm(d1, d2, eris, time):
    foo,fov,fvo,fvv = utils.compute_comm(d1, d2, eris, time)
    fd, fd_ = eris.fd
    Bov = fov - fvo.T.conj()
    Bvo =  - Bov.T.conj()
    Boo = foo - foo.T.conj()
    Bvv = fvv - fvv.T.conj()
    B  = einsum('qp,q,p->qp',Bov,fd,fd_)
    B += einsum('qp,q,p->qp',Bvo,fd_,fd)
    B += einsum('qp,p,q->qp',Boo,fd,fd) 
    B += einsum('qp,p,q->qp',Bvv,fd_,fd_) 
    Bov = Bvo = Boo = Bvv = None
    fov = fvo = foo = fvv = None
    return B

def kernel(mf, t, l, w, f0, td, tf, step, thresh=1e-3):
    nmo = mf.mol.nao_nr()
    N = int((tf+step*0.1)/step)
    C = np.eye(nmo, dtype=complex) # in physical basis

    eris = ERIs(mf, w, f0, td) # in HF basis
    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
#    d1, d2 = utils.compute_rdm12(t, l)
#    e = utils.compute_energy(d1, d2, eris, time=None)
#    print('check initial energy: {}'.format(e.real+mf.energy_nuc())) 

    fd, fd_ = eris.fd
    doo, dvv = utils.compute_rdm1(t, l)
    d1_old  = einsum('pq,p,q->pq',doo,fd ,fd )
    d1_old += einsum('pq,p,q->pq',dvv,fd_,fd_)
    E = np.zeros(N+1,dtype=complex) 
#    mu = np.zeros((N+1,3),dtype=complex)  
    for i in range(N+1):
        time = i * step
        eris.rotate(C)
        d1, d2 = utils.compute_rdm12(t, l)
        X, Xb = compute_X(d1, d2, eris, time, thresh) # in physical basis 
        dt, dl = utils.update_amps(t, l, eris, time, Xb)
        E[i] = utils.compute_energy(d1, d2, eris, time=None) # <U^{-1}H0U>
#        mu[i,:] = einsum('qp,xpq->x',utils.rotate1(d1,C.T.conj()),eris.mu_) 
        # update 
        t += step * dt
        l += step * dl
        C = np.dot(scipy.linalg.expm(-step*X), C)
        # Ehrenfest error
        d1_new, d2_new = utils.compute_rdm12(t, l)
        B = compute_comm(d1_new, d2_new, eris, time) # B_qp=<[H_U,p+q]>
        doo, dvv = d1_new
        d1_new  = einsum('pq,p,q->pq',doo,fd ,fd )
        d1_new += einsum('pq,p,q->pq',dvv,fd_,fd_)
        d1_new = utils.rotate1(d1_new, C.T.conj())
        B = utils.rotate1(B, C.T.conj())
        err = np.linalg.norm((d1_new-d1_old)/step-1j*B)
        d1_old = d1_new.copy()
        print('time: {:.4f}, EE(mH): {}, X: {}, err: {}'.format(
              time, (E[i] - E[0]).real*1e3, np.linalg.norm(X)**2, err))
#    print('trace error: ',tr)
#    print('energy conservation error: ', ec)
    print('imaginary part of energy: ', np.linalg.norm(E.imag))
#    return (E - E[0]).real, (mu - eris.nucl_dip).real
#    return d1_new, 1j*B, C, X, t, l

class ERIs:
    def __init__(self, mf, w=0.0, f0=np.zeros(3), td=0.0, beta=0.0, mu=0.0):
        self.no = mf.mol.nelec[0]
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

        mo_coeff = mf.mo_coeff.copy()
        self.h0_ = einsum('uv,up,vq->pq',hao,mo_coeff,mo_coeff)
        self.h1_ = einsum('uv,up,vq->pq',h1ao,mo_coeff,mo_coeff)
        self.mu_ = einsum('xuv,up,vq->xpq',mu_ao,mo_coeff,mo_coeff) # dipole
        self.eri_ = einsum('uvxy,up,vr->prxy',eri_ao,mo_coeff,mo_coeff)
        self.eri_ = einsum('prxy,xq,ys->prqs',self.eri_,mo_coeff,mo_coeff)
        self.eri_ = self.eri_.transpose(0,2,1,3)

        self.h0 = self.h0_.copy()
        self.h1 = self.h1_.copy()
        self.eri = self.eri_.copy()

        muao = h1ao = eri_ao = None

    def rotate(self, C, time=None): # rotate ZT integrals
        self.h0 = utils.rotate1(self.h0_, C)
        self.h1 = utils.rotate1(self.h1_, C)
        self.eri = utils.rotate2(self.eri_, C)

    def make_tensors(self, time=None):
        h = utils.full_h(self.h0, self.h1, self.w, self.td, time) 
        fd, fd_ = self.fd

        self.hoo = einsum('pq,p,q->pq',h,fd ,fd )
        self.hvv = einsum('pq,p,q->pq',h,fd_,fd_)
        self.hov = einsum('pq,p,q->pq',h,fd ,fd_)
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

        h = None

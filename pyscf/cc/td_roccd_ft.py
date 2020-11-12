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
    no = len(fd)
    d1_  = einsum('pq,p,q->pq',d1[:no,:no],fd ,fd ) 
    d1_ += einsum('pq,p,q->pq',d1[:no,no:],fd ,fd_) 
    d1_ += einsum('pq,p,q->pq',d1[no:,:no],fd_,fd ) 
    d1_ += einsum('pq,p,q->pq',d1[no:,no:],fd_,fd_) 
    return d1_

def kernel(eris, t, l, tf, step, RK=4, orb=True):
    no, _, nv, _ = l.shape
    nmo = no + nv
    N = int((tf+step*0.1)/step)
    if not eris.phys: # for Bogoliubov rotation
        C = np.eye(nmo, dtype=complex)
    else: # for physical rotation
        C = np.eye(no, dtype=complex)

    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    d1, d2 = utils.compute_rdm12(t, l)
    e = utils.compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+eris.mf.energy_nuc())) 

    d1_old = np.block([[d1[0],np.zeros((no,nv))],
                       [np.zeros((nv,no)),d1[1]]])
    d1_old = compute_phys1(d1_old, eris)
    E = np.zeros(N+1,dtype=complex) 
    rdm1 = np.zeros((N+1,no,no),dtype=complex) 
    for i in range(N+1):
        time = i * step
#        X, _ = compute_X(d1, d2, eris, time)
#        dt, dl = utils.update_amps(t, l, eris, time)
        dt, dl, X, E[i], F = utils.update_RK(t, l, C, eris, time, step, RK, orb) # <H_U>
#        mu[i,:] = einsum('qp,xpq->x',utils.rotate1(d1,C.T.conj()),eris.mu_) 
        # update 
        t += step * dt
        l += step * dl
        C = np.dot(scipy.linalg.expm(-step*X), C)
        # Ehrenfest error
        d1 = utils.compute_rdm1(t, l)
        d1_new = np.block([[d1[0],np.zeros((no,nv))],
                           [np.zeros((nv,no)),d1[1]]])
        if RK == 1:
            if not eris.phys:
                d1_new = utils.rotate1(d1_new, C.T.conj())
                d1_new = compute_phys1(d1_new, eris)
                F = np.block([[F[0],F[1]],[F[2],F[3]]])
                F -= F.T.conj()
                F = utils.rotate1(F, C.T.conj())
                F = compute_phys1(F, eris)
            else:
                d1_new = compute_phys1(d1_new, eris)
                d1_new = utils.rotate1(d1_new, C.T.conj())
                F = np.block([[F[0],F[1]],[F[2],F[3]]])
                F -= F.T.conj()
                F = compute_phys1(F, eris)
                F = utils.rotate1(F, C.T.conj())
            err = np.linalg.norm((d1_new-d1_old)/step-1j*F)
            print(abs(np.trace(F)))
            d1_old = d1_new.copy()
            print('time: {:.4f}, EE(mH): {}, X: {}, err: {}'.format(
                  time, (E[i] - E[0]).real*1e3, np.linalg.norm(X), err))
        else: 
            d1_new = utils.rotate1(d1_new, C.T.conj())
            d1_new = compute_phys1(d1_new, eris)
            tr = 2.0*np.trace(d1_new)
            print('time: {:.4f}, EE(mH): {}, X: {}, tr: {}'.format(
                  time, (E[i] - E[0]).real*1e3, np.linalg.norm(X), tr.real))
        rdm1[i,:,:] = d1_new.copy()
    return rdm1

class ERIs_mol:
    def __init__(self, mf, f0=np.zeros(3), w=0.0, td=0.0, 
                 beta=0.0, mu=0.0, picture='I'):
        self.mf = mf
        self.w = w
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.picture = picture
        self.fd = compute_sqrt_fd(mf.mo_energy, beta, mu)
        self.phys = False

        # integrals in fixed Bogliubov basis
        fd, fd_ = self.fd
        no = mf.mol.nao_nr()
        h0, h1, eri = utils.mo_ints_mol(mf, f0)[:3]
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
        h0 = h1 = mu = eri = None

    def rotate(self, C, time=None):
        self.h0 = utils.rotate1(self.h0_, C)
        self.h1 = utils.rotate1(self.h1_, C)
        self.eri = utils.rotate2(self.eri_, C)

    def make_tensors(self, time=None):
        no = self.mf.mol.nao_nr()
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
        self.vovv = self.eri[no:,:no,no:,no:].copy()
        self.oovo = self.eri[:no,:no,no:,:no].copy()
        self.ooov = self.eri[:no,:no,:no,no:].copy()

        self.foo  = self.hoo.copy()
        self.foo += 2.0 * einsum('ikjk->ij',self.oooo)
        self.foo -= einsum('ikkj->ij',self.oooo)
        self.fvv  = self.hvv.copy()
        self.fvv += 2.0 * einsum('kakb->ab',self.ovov)
        self.fvv -= einsum('kabk->ab',self.ovvo)

        if self.picture == 'I':
            fd, fd_ = self.fd
            tmp_oo = einsum('pq,p,q->pq',np.diag(self.mf.mo_energy),fd, fd ) 
            tmp_vv = einsum('pq,p,q->pq',np.diag(self.mf.mo_energy),fd_,fd_) 
            tmp_oo -= np.diag(self.mf.mo_energy)
            tmp_vv -= np.diag(self.mf.mo_energy)
            self.foo -= tmp_oo
            self.fvv -= tmp_vv
            self.hoo -= tmp_oo
            self.hvv -= tmp_vv
        h = None

class ERIs_sol:
    def __init__(self, mf, f0=np.zeros(3), sigma=1.0, w=0.0, td=0.0, 
                 beta=0.0, mu=0.0, picture='I'):
        self.mf = mf
        self.sigma = sigma
        self.w = w
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.picture = picture
        self.fd = compute_sqrt_fd(mf.mo_energy, beta, mu)
        self.phys = False

        # integrals in fixed Bogliubov basis
        fd, fd_ = self.fd
        no = mf.mol.nao_nr()
        h0, h1, eri = utils.mo_ints_cell(mf, f0)[:3]
        self.h0_ = np.zeros((no*2,)*2, dtype=complex)
        self.h1_ = np.zeros((no*2,)*2, dtype=complex)
        self.eri_ = np.zeros((no*2,)*4, dtype=complex)
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
        h0 = h1 = mu = eri = None

    def rotate(self, C, time=None):
        self.h0 = utils.rotate1(self.h0_, C)
        self.h1 = utils.rotate1(self.h1_, C)
        self.eri = utils.rotate2(self.eri_, C)

    def make_tensors(self, time=None):
        no = self.mf.cell.nao_nr()
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
        self.vovv = self.eri[no:,:no,no:,no:].copy()
        self.oovo = self.eri[:no,:no,no:,:no].copy()
        self.ooov = self.eri[:no,:no,:no,no:].copy()

        self.foo  = self.hoo.copy()
        self.foo += 2.0 * einsum('ikjk->ij',self.oooo)
        self.foo -= einsum('ikkj->ij',self.oooo)
        self.fvv  = self.hvv.copy()
        self.fvv += 2.0 * einsum('kakb->ab',self.ovov)
        self.fvv -= einsum('kabk->ab',self.ovvo)

        if self.picture == 'I':
            fd, fd_ = self.fd
            tmp_oo = einsum('pq,p,q->pq',np.diag(self.mf.mo_energy),fd, fd ) 
            tmp_vv = einsum('pq,p,q->pq',np.diag(self.mf.mo_energy),fd_,fd_) 
            tmp_oo -= np.diag(self.mf.mo_energy)
            tmp_vv -= np.diag(self.mf.mo_energy)
            self.foo -= tmp_oo
            self.fvv -= tmp_vv
            self.hoo -= tmp_oo
            self.hvv -= tmp_vv
        h = None

class ERIs_p:
    def __init__(self, mf, f0=np.zeros(3), w=0.0, td=0.0, 
                 beta=0.0, mu=0.0, picture='I'):
        self.mf = mf
        self.w = w
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.picture = picture
        self.fd = compute_sqrt_fd(mf.mo_energy, beta, mu)
        self.phys = True
        # ZT integrals in HF basis
        self.h0_, self.h1_, self.eri_ = utils.mo_ints_mol(mf, f0)[:3]
        # ZT integrals in rotating basis
        self.h0 = np.array(self.h0_, dtype=complex)
        self.h1 = np.array(self.h1_, dtype=complex)
        self.eri = np.array(self.eri_, dtype=complex)

    def rotate(self, C): # rotate ZT integrals
        self.h0 = utils.rotate1(self.h0_, C)
        self.h1 = utils.rotate1(self.h1_, C)
        self.eri = utils.rotate2(self.eri_, C)

    def make_tensors(self, time=None):
        self.h = self.h0.copy()
        if time is not None: 
            self.h += self.h1 * utils.fac_mol(self.w, self.td, time) 
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
#        self.foo  = self.hoo.copy()
#        self.foo += 2.0 * einsum('ikjk->ij',self.oooo)
#        self.foo -= einsum('ikkj->ij',self.oooo)
#        self.fvv  = self.hvv.copy()
#        self.fvv += 2.0 * einsum('kakb->ab',self.ovov)
#        self.fvv -= einsum('kabk->ab',self.ovvo)
        f = self.h.copy()
        f += 2.0 * einsum('prqr,r->pq',self.eri,np.square(fd))
        f -= einsum('prrq,r->pq',self.eri,np.square(fd))
        self.foo = einsum('pq,p,q->pq',f,fd ,fd )
        self.fvv = einsum('pq,p,q->pq',f,fd_,fd_)

        if self.picture == 'I': 
            tmp_oo = einsum('pq,p,q->pq',np.diag(self.mf.mo_energy),fd, fd ) 
            tmp_vv = einsum('pq,p,q->pq',np.diag(self.mf.mo_energy),fd_,fd_) 
            tmp_oo -= np.diag(self.mf.mo_energy)
            tmp_vv -= np.diag(self.mf.mo_energy)
            self.foo -= tmp_oo
            self.fvv -= tmp_vv
            self.hoo -= tmp_oo
            self.hvv -= tmp_vv
        h = None

    def compute_rdm1(self, t, l, dt=None, dl=None):
        fd, fd_ = self.fd
        doo, dvv = utils.compute_rdm1(t, l, dt, dl)
        d1  = einsum('pq,p,q->pq',doo,fd ,fd ) 
        d1 += einsum('pq,p,q->pq',dvv,fd_,fd_) 
        return d1

    def compute_rdm12(self, t, l):
        fd, fd_ = self.fd
        (doo, dvv), (doooo, doovv, dovvo, dovov, dvvvv) = utils.compute_rdm12(t, l)
        d1  = einsum('pq,p,q->pq',doo,fd ,fd ) 
        d1 += einsum('pq,p,q->pq',dvv,fd_,fd_) 
        d2  = einsum('pqrs,p,q,r,s->pqrs',doooo,fd ,fd ,fd ,fd ) # oooo 
        d2 += einsum('pqrs,p,q,r,s->pqrs',doovv,fd ,fd ,fd_,fd_) # oovv
        d2 += einsum('pqrs,p,q,r,s->rspq',doovv.conj(),fd ,fd ,fd_,fd_) # vvoo
        d2 += einsum('pqrs,p,q,r,s->pqrs',dovvo,fd ,fd_,fd_,fd ) # ovvo
        d2 += einsum('pqrs,p,q,r,s->qpsr',dovvo,fd ,fd_,fd_,fd ) # voov
        d2 += einsum('pqrs,p,q,r,s->pqrs',dovov,fd ,fd_,fd ,fd_) # ovov
        d2 += einsum('pqrs,p,q,r,s->qpsr',dovov,fd ,fd_,fd ,fd_) # vovo
        d2 += einsum('pqrs,p,q,r,s->qpsr',dvvvv,fd_,fd_,fd_,fd_) # vvvv
        return d1, d2

    def compute_comm(self, d1, d2, time=None): #f_qp = <[H,p+q]>
        self.make_tensors(time)
        eri_ = self.eri - self.eri.transpose(0,1,3,2)
        d2_ = d2 - d2.transpose(0,1,3,2)
    
        f  = einsum('vp,pu->vu',d1,self.h)
        f += 0.5 * einsum('vspq,pqus->vu',d2_,eri_)
        f += einsum('vspq,pqus->vu',d2,self.eri)
        return f

    def compute_energy(self, d1, d2, time=None):
        self.make_tensors(time)
        eri_ = self.eri - self.eri.transpose(0,1,3,2)
        d2_ = d2 - d2.transpose(0,1,3,2)
        
        e  = 2.0 * einsum('pq,qp',self.h,d1)
        e += 0.5 * einsum('pqrs,rspq',eri_,d2_)
        e += einsum('pqrs,rspq',self.eri,d2)
        return e.real 


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


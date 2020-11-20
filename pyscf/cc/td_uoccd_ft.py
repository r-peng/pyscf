import numpy as np
from pyscf import lib, cc
from pyscf.cc import td_uoccd_utils as utils
import scipy
einsum = lib.einsum

class ERIs_mol:
    def __init__(self, mf, f0=np.zeros(3), w=0.0, td=0.0, 
                 beta=0.0, mu=0.0, picture='I'):
        self.mf = mf
        self.w = w
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.picture = picture
        self.fd = utils.compute_sqrt_fd(mf.mo_energy, beta, mu)

        # integrals in fixed Bogliubov basis
        (fda, fda_), (fdb, fdb_) = self.fd
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

        if picture == 'I':
            self.Roo = np.diag(mf.mo_energy)
            self.Rvv = np.diag(mf.mo_energy)
            self.Roo -= einsum('pq,p,q->pq',np.diag(mf.mo_energy),fd, fd ) 
            self.Rvv -= einsum('pq,p,q->pq',np.diag(mf.mo_energy),fd_,fd_) 

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
            self.foo += self.Roo
            self.fvv += self.Rvv
        h = None



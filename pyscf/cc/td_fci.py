from pyscf.fci import direct_spin1, cistring
from pyscf import lib, cc
from pyscf.cc import td_occd_slow
import numpy as np
import math
einsum = lib.einsum

def update_ci(c, eris):
    h2e = direct_spin1.absorb_h1e(eris.h, eris.eri, eris.nao, eris.nelec, 0.5)
    def _hop(c):
        return direct_spin1.contract_2e(h2e, c, eris.nao, eris.nelec)
    Hc = _hop(c)
    Hc -= eris.E0
    return - Hc

def update_RK4(c, eris, step, RK4=True):
    dc1 = update_ci(c, eris)
    if not RK4:
        return dc1
    else: 
        dc2 = update_ci(c+dc1*step*0.5, eris)
        dc3 = update_ci(c+dc2*step*0.5, eris)
        dc4 = update_ci(c+dc3*step    , eris)
        return (dc1+2.0*dc2+2.0*dc3+dc4)/6.0

def compute_energy(d1, d2, eris, time=None):
    eris.full_h(time)
    eri_ab = eris.eri.transpose(0,2,1,3).copy()
    eri_aa = eri_ab - eri_ab.transpose(0,1,3,2)
    e  = einsum('pq,qp',eris.h,d1[0])
    e += einsum('PQ,QP',eris.h,d1[1])
    e += 0.25 * einsum('pqrs,rspq',eri_aa,d2[0])
    e += 0.25 * einsum('PQRS,RSPQ',eri_aa,d2[2])
    e +=        einsum('pQrS,rSpQ',eri_ab,d2[1])
    return e

def compute_rdm(c, nao, nelec):
    d1, d2 = direct_spin1.make_rdm12s(c, nao, nelec)
    d2aa = d2[0].transpose(1,3,0,2) 
    d2ab = d2[1].transpose(1,3,0,2)
    d2bb = d2[2].transpose(1,3,0,2)
    return d1, (d2aa, d2ab, d2bb)

def compute_trace(d1, d2, eris):
    tr1  = abs(np.trace(d1[0])-eris.nelec[0])
    tr1 += abs(np.trace(d1[1])-eris.nelec[1])
    d2_  = einsum('prqr->pq',d2[0])
    d2_ += einsum('prqr->pq',d2[1])
    d2_ += einsum('prqr->pq',d2[2])
    d2_ += einsum('rprq->pq',d2[1])
    d2_ /= sum(eris.nelec) - 1
    tr2 = np.linalg.norm(d2_-d1[0]-d1[1])
    return tr1, tr2

def kernel_it(mf, step=0.01, maxiter=1000, thresh=1e-6, RK4=True):
    eris = ERIs(mf)
    eris.full_h() 

    N = cistring.num_strings(eris.nao, eris.nelec[0])
    c = np.zeros((N,N))
    c[0,0] = 1.0
    E_old = mf.energy_elec()[0]
    for i in range(maxiter):
        dc = update_RK4(c, eris, step, RK4=RK4)
        c += step * dc
        c /= np.linalg.norm(c)
        d1, d2 = compute_rdm(c, eris.nao, eris.nelec)
        E = compute_energy(d1, d2, eris)
        dnorm = np.linalg.norm(dc)
        dE, E_old = E-E_old, E
        print('iter: {}, dnorm: {}, dE: {}, E: {}'.format(i, dnorm, dE, E))
        if abs(dE) < thresh:
            break
    return c, E

def kernel_rt_test(mf, c, w, f0, td, ts, RK4=True):
    eris = ERIs(mf, w=w, f0=f0, td=td)
    eris.full_h()
    c = CI(c)

    N = len(ts)
    d1as = np.zeros((N,eris.nao,eris.nao),dtype=complex)  
    d1bs = np.zeros((N,eris.nao,eris.nao),dtype=complex)  
    E = np.zeros(N,dtype=complex)

    d1, d2 = c.compute_rdm(eris.nao, eris.nelec, compute_d2=True)
    d1as[0,:,:], d1bs[0,:,:] = d1[0].copy(), d1[1].copy()
    E[0] = compute_energy(d1, d2, eris, time=None)
    tr = compute_trace(d1, d2, eris)
    print('check trace: {}'.format(tr))
    for i in range(1,N):
        time = ts[i]
        step = ts[i] - ts[i-1]
        dr, di = c.update_RK4(c.real, c.imag, eris, time, step, RK4=RK4)
        # computing observables
        d1, d2 = c.compute_rdm(eris.nao, eris.nelec, compute_d2=True)
        d1as[i,:,:], d1bs[i,:,:] = d1[0].copy(), d1[1].copy()
        LHS = (d1as[i]-d1as[i-1])/step, (d1bs[i]-d1bs[i-1])/step
        RHS = compute_RHS(d1, d2, eris, time)
        err = np.linalg.norm(LHS[0]-RHS[0]), np.linalg.norm(LHS[1]-RHS[1])
        E[i] = compute_energy(d1, d2, eris, time)
        tr = compute_trace(d1, d2, eris)
        print('time: {:.4f}, ee(mH): {}, d1: {}, tr: {}'.format(
               time, (E[i]-E[0]).real*1e3, err, tr))
        if sum(err) > 1.0:
            print('diverging error!')
            break
        # update CI
        c.real += step * dr
        c.imag += step * di
        norm = np.linalg.norm(c.real+1j*c.imag)
        c.real /= norm 
        c.imag /= norm

def kernel_rt(mf, c, w, f0, td, ts, RK4=True):
    eris = ERIs(mf, w=w, f0=f0, td=td)
    eris.full_h()
    c = CI(c)

    N = len(ts)
    mus = np.zeros((N,3),dtype=complex)
    Hmu = np.zeros((N,3),dtype=complex)
    E = np.zeros(N,dtype=complex)

    d1, d2 = c.compute_rdm(eris.nao, eris.nelec, compute_d2=True)
    mus[0,:], _ = compute_dipole(eris, d1, d2=None, time=None) 
    E[0] = compute_energy(d1, d2, eris, time=None)
    print('FCI energy check : {}'.format(E[0]+mf.energy_nuc()))
    for i in range(1,N):
        time = ts[i]
        step = ts[i] - ts[i-1]
        dr, di = c.update_RK4(c.real, c.imag, eris, time, step, RK4=RK4)
        # computing observables
        d1, d2 = c.compute_rdm(eris.nao, eris.nelec, compute_d2=True)
        mus[i,:], Hmu[i,:] = compute_dipole(eris, d1, d2, time) 
        err = (mus[i,:]-mus[i-1,:])/step - Hmu[i,:]
        E[i] = compute_energy(d1, d2, eris, time=None)
        print('time: {:.4f}, E(mH): {}, err: {}, mu.imag: {}, E.imag: {}'.format(
               time, (E[i]-E[0]).real*1e3, abs(sum(err)), sum(mus[i,:]).imag , E[i].imag))
        # update CI
        c.real += step * dr
        c.imag += step * di
        norm = np.linalg.norm(c.real+1j*c.imag)
        c.real /= norm 
        c.imag /= norm
    return mus.real, (E - E[0]).real

class ERIs():
    def __init__(self, mf, w=None, f0=None, td=None):
        hao = mf.get_hcore()
        eri_ao = mf.mol.intor('int2e_sph')
        self.E0 = 0.0
        self.nao = mf.mol.nao_nr()
        self.nelec = mf.mol.nelec

        self.h0 = einsum('uv,up,vq->pq',hao,mf.mo_coeff.conj(),mf.mo_coeff)
        eri = einsum('uvxy,up,vr->prxy',eri_ao,mf.mo_coeff.conj(),mf.mo_coeff)
        self.eri = einsum('prxy,xq,ys->prqs',eri,mf.mo_coeff.conj(),mf.mo_coeff)

        if td is not None: 
            mu_ao = mf.mol.intor('int1e_r')
            h1ao = einsum('xuv,x->uv',mu_ao,f0)
            self.h1 = einsum('uv,up,vq->pq',h1ao,mf.mo_coeff.conj(),mf.mo_coeff)
            self.mu = einsum('xuv,up,vq->xpq',mu_ao,mf.mo_coeff.conj(),mf.mo_coeff)
            self.w = w
            self.f0 = f0
            self.td = td

    def full_h(self, time=None):
        self.h = self.h0.copy()
        if time is not None: 
            if time < self.td:
                #print('td')
                evlp = math.sin(math.pi*time/self.td)**2
                #osc = math.sin(self.w*time)
                osc = math.cos(self.w*(time-self.td*0.5))
                self.h += self.h1 * evlp * osc

class CI():
    def __init__(self, c):
        self.real = c.copy()
        self.imag = np.zeros_like(c)

    def compute_rdm(self, norb, nelec, compute_d2=True):
        # direct_spin returns: 
        # 1pdm[p,q] = :math:`\langle q^\dagger p\rangle
        # 2pdm[p,q,r,s] = \langle p^\dagger r^\dagger s q\rangle
        if compute_d2: 
            rr1, rr2 = direct_spin1.make_rdm12s(self.real, norb, nelec)
            ii1, ii2 = direct_spin1.make_rdm12s(self.imag, norb, nelec)
            ri1, ri2 = direct_spin1.trans_rdm12s(self.real, self.imag, norb, nelec)
            d2aa = rr2[0] + ii2[0] + 1j * ri2[0] - 1j * ri2[0].transpose(1,0,3,2)
            d2ab = rr2[1] + ii2[1] + 1j * ri2[1] - 1j * ri2[2].transpose(3,2,1,0)
            d2bb = rr2[2] + ii2[2] + 1j * ri2[3] - 1j * ri2[3].transpose(1,0,3,2)
            # transpose so that 2pdm[q,s,p,r] = \langle p^\dagger r^\dagger s q\rangle
            d2aa = d2aa.transpose(1,3,0,2) 
            d2ab = d2ab.transpose(1,3,0,2)
            d2bb = d2bb.transpose(1,3,0,2)
        else: 
            rr1 = direct_spin1.make_rdm1s(self.real, norb, nelec)
            ii1 = direct_spin1.make_rdm1s(self.imag, norb, nelec)
            ri1 = direct_spin1.trans_rdm1s(self.real, self.imag, norb, nelec)
            d2aa = d2ab = d2bb = None
        d1a = rr1[0] + ii1[0] + 1j * ri1[0] - 1j * ri1[0].T
        d1b = rr1[1] + ii1[1] + 1j * ri1[1] - 1j * ri1[1].T
        return (d1a, d1b), (d2aa, d2ab, d2bb)

    def update_ci(self, real, imag, eris):
        dr = update_ci(imag, eris) * (-1.0)
        di = update_ci(real, eris)
        return dr, di

    def update_RK4(self, r, i, eris, time, step, RK4=True):
        eris.full_h(time)
        dr1, di1 = self.update_ci(r, i, eris)
        if not RK4:
            return dr1, di1
        else: 
            eris.full_h(time+step*0.5)
            dr2, di2 = self.update_ci(r+dr1*step*0.5, i+di1*step*0.5, eris) 
            dr3, di3 = self.update_ci(r+dr2*step*0.5, i+di2*step*0.5, eris)
            eris.full_h(time+step)
            dr4, di4 = self.update_ci(r+dr3*step    , i+di3*step    , eris)
            dr = (dr1+2.0*dr2+2.0*dr3+dr4)/6.0
            di = (di1+2.0*di2+2.0*di3+di4)/6.0
            return dr, di
    
def compute_RHS(d1, d2, eris, time=None):
    eris.full_h(time)
    eri_ab = eris.eri.transpose(0,2,1,3).copy()
    eri_aa = eri_ab - eri_ab.transpose(0,1,3,2)
    def compute_imd(da, daa, dab, eriaa, eriab):
        C  = einsum('vp,pu->uv',da,eris.h)
        C += 0.5 * einsum('pqus,vspq->uv',eriaa,daa)
        C +=       einsum('pQuS,vSpQ->uv',eriab,dab)
        return C - C.T.conj()
    Ca = compute_imd(d1[0], d2[0], d2[1], eri_aa, eri_ab)
    Cb = compute_imd(d1[1], d2[2], d2[1].transpose(1,0,3,2), 
                     eri_aa, eri_ab.transpose(1,0,3,2))
    return 1j*Ca.T, 1j*Cb.T

def compute_dipole(eris, d1, d2=None, time=None):
    mu  = einsum('xpq,qp->x',eris.mu,d1[0])
    mu += einsum('xpq,qp->x',eris.mu,d1[1])
    Hmu = None
    if d2 is not None: 
        RHS = compute_RHS(d1, d2, eris, time)
        Hmu  = einsum('xpq,qp->x',eris.mu,RHS[0])
        Hmu += einsum('xpq,qp->x',eris.mu,RHS[1])
    return mu, Hmu


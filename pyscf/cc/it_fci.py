from pyscf.fci import direct_spin1, direct_uhf, cistring
from pyscf import lib
import numpy as np
import math
einsum = lib.einsum

def update_ci(c, h1e, eri, E0, nelec):
    nmo = h1e.shape[0]
    h2e = direct_spin1.absorb_h1e(h1e, eri, nmo, nelec, 0.5)
    def _hop(c):
        return direct_spin1.contract_2e(h2e, c, nmo, nelec)
    Hc = _hop(c)
    Hc -= E0
    return - Hc

def update_RK4(c, h1e, eri, E0, nelec, step, RK4=True):
    dc1 = update_ci(c, h1e, eri, E0, nelec)
    if not RK4:
        return dc1
    else: 
        dc2 = update_ci(c+dc1*step*0.5, h1e, eri, E0, nelec)
        dc3 = update_ci(c+dc2*step*0.5, h1e, eri, E0, nelec)
        dc4 = update_ci(c+dc3*step    , h1e, eri, E0, nelec)
        return (dc1+2.0*dc2+2.0*dc3+dc4)/6.0

def compute_energy(d1, d2, h1e, eri):
    d2aa, d2ab, d2bb = d2
    d2aa = d2aa.transpose(0,2,1,3)
    d2ab = d2ab.transpose(0,2,1,3)
    d2bb = d2bb.transpose(0,2,1,3)
    eri_ab = eri.transpose(0,2,1,3)
    eri_aa = eri_ab - eri_ab.transpose(0,1,3,2)
    e  = einsum('pq,qp',h1e,d1[0])
    e += einsum('PQ,QP',h1e,d1[1])
    e += 0.25 * einsum('pqrs,rspq',eri_aa,d2aa)
    e += 0.25 * einsum('PQRS,RSPQ',eri_aa,d2bb)
    e +=        einsum('pQrS,rSpQ',eri_ab,d2ab)
    return e

def kernel_it(mf, step=0.01, maxiter=1000, thresh=1e-6, RK4=True):
    nao = mf.mol.nao_nr()
    nelec = mf.mol.nelec
    E0 = mf.energy_elec()[0]
    E0 = 0.0
    Enuc = mf.energy_nuc()

    hao = mf.get_hcore()
    eri_ao = mf.mol.intor('int2e_sph')
    
    h = einsum('uv,up,vq->pq',hao,mf.mo_coeff.conj(),mf.mo_coeff)
    eri = einsum('uvxy,up,vr->prxy',eri_ao,mf.mo_coeff.conj(),mf.mo_coeff)
    eri = einsum('prxy,xq,ys->prqs',eri   ,mf.mo_coeff.conj(),mf.mo_coeff)

    N = cistring.num_strings(nao, nelec[0])
    c = np.zeros((N,N))
    c[0,0] = 1.0
    E_old = mf.energy_elec()[0]
    d1, d2 = direct_spin1.make_rdm12s(c, nao, nelec)
    for i in range(maxiter):
        dc = update_RK4(c, h, eri, E0, nelec, step, RK4=RK4)
        c += step * dc
        c /= np.linalg.norm(c)
        d1, d2 = direct_spin1.make_rdm12s(c, nao, nelec)
        E = compute_energy(d1, d2, h, eri)
        dnorm = np.linalg.norm(dc)
        dE, E_old = E-E_old, E
        print('iter: {}, dnorm: {}, dE: {}, E: {}'.format(i, dnorm, dE, E))
        if abs(dE) < thresh:
            break
    return c, E


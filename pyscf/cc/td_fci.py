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

def kernel_rt_test(mf, c, w, f0, tp, tf, step, RK4=True):
    nao = mf.mol.nao_nr()
    nelec = mf.mol.nelec
    E0 = 0.0

    hao = mf.get_hcore()
    eri_ao = mf.mol.intor('int2e_sph')
    mu_ao = mf.mol.intor('int1e_r')
    hao_td = einsum('xuv,x->uv',mu_ao,f0)

    h = einsum('uv,up,vq->pq',hao,mf.mo_coeff.conj(),mf.mo_coeff)
    htd = einsum('uv,up,vq->pq',hao_td,mf.mo_coeff.conj(),mf.mo_coeff)
    eri = einsum('uvxy,up,vr->prxy',eri_ao,mf.mo_coeff.conj(),mf.mo_coeff)
    eri = einsum('prxy,xq,ys->prqs',eri   ,mf.mo_coeff.conj(),mf.mo_coeff)

    td = 2 * int(tp/step)
    maxiter = int(tf/step)+1
    c = CI(c)

    d1_old, _ = c.compute_rdm(nao, nelec, compute_d2=False)
    tr  = abs(np.trace(d1_old[0])-nelec[0])
    tr += abs(np.trace(d1_old[1])-nelec[1])
    print('check trace: {}'.format(tr))
#    exit()
    for i in range(maxiter):
        h1e = h.copy()
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
            osc = math.sin(w*i*step) 
            h1e += htd.copy() * osc * evlp
        dr, di = c.update_RK4(c.real, c.imag, h1e, eri, E0, nelec, step, RK4=RK4)
        c.real += step * dr
        c.imag += step * di
        c.real /= np.linalg.norm(c.real+1j*c.imag)
        c.imag /= np.linalg.norm(c.real+1j*c.imag)
        d1, d2 = c.compute_rdm(nao, nelec, compute_d2=True)
        dd1, d1_old = (d1[0]-d1_old[0], d1[1]-d1_old[1]), d1
        LHS = dd1[0]/step, dd1[1]/step
        RHS = c.compute_RHS(d1, d2, h1e, eri)
        error = np.linalg.norm(LHS[0]-RHS[0]), np.linalg.norm(LHS[1]-RHS[1])
        tr  = abs(np.trace(d1_old[0])-nelec[0])
        tr += abs(np.trace(d1_old[1])-nelec[1])
        print('time: {:.4f}, d1a: {}, d1b: {}, tr: {}'.format(
               i*step, error[0], error[1], tr))
        if sum(error) > 1.0:
            print('diverging error!')
            break
    print('check trace: {}'.format(tr))

def kernel_rt(mf, c, w, f0, tp, tf, step, RK4=True):
    nao = mf.mol.nao_nr()
    nelec = mf.mol.nelec
    E0 = 0.0

    hao = mf.get_hcore()
    eri_ao = mf.mol.intor('int2e_sph')
    mu_ao = mf.mol.intor('int1e_r')
    hao_td = einsum('xuv,x->uv',mu_ao,f0)

    h = einsum('uv,up,vq->pq',hao,mf.mo_coeff.conj(),mf.mo_coeff)
    mu_mo = einsum('xuv,up,vq->xpq',mu_ao,mf.mo_coeff.conj(),mf.mo_coeff)
    htd = einsum('uv,up,vq->pq',hao_td,mf.mo_coeff.conj(),mf.mo_coeff)
    eri = einsum('uvxy,up,vr->prxy',eri_ao,mf.mo_coeff.conj(),mf.mo_coeff)
    eri = einsum('prxy,xq,ys->prqs',eri   ,mf.mo_coeff.conj(),mf.mo_coeff)

    td = 2 * int(tp/step)
    maxiter = int(tf/step)+1
    c = CI(c)

    mu = np.zeros((3,maxiter+1),dtype=complex)
    Hmu = np.zeros((3,maxiter),dtype=complex)
    E = np.zeros(maxiter,dtype=complex)

    d1, _ = c.compute_rdm(nao, nelec, compute_d2=False)
    mu[:,0]  = einsum('xpq,qp->x',mu_mo,d1[0])
    mu[:,0] += einsum('xpq,qp->x',mu_mo,d1[1])
    for i in range(maxiter):
        h1e = h.copy()
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
            osc = math.sin(w*i*step) 
            h1e += htd.copy() * osc * evlp
        dr, di = c.update_RK4(c.real, c.imag, h1e, eri, E0, nelec, step, RK4=RK4)
        c.real += step * dr
        c.imag += step * di
        c.real /= np.linalg.norm(c.real+1j*c.imag)
        c.imag /= np.linalg.norm(c.real+1j*c.imag)
        d1, d2 = c.compute_rdm(nao, nelec, compute_d2=True)
        mu[:,i+1]  = einsum('xpq,qp->x',mu_mo,d1[0])
        mu[:,i+1] += einsum('xpq,qp->x',mu_mo,d1[1])
        E[i] = compute_energy(d1, d2, h1e, eri)
        RHS = c.compute_RHS(d1, d2, h1e, eri)
        Hmu[:,i]  = einsum('xpq,qp->x',mu_mo,RHS[0])
        Hmu[:,i] += einsum('xpq,qp->x',mu_mo,RHS[1])
        error = (mu[:,i+1]-mu[:,i])/step - Hmu[:,i]
        imag = sum(mu[:,i+1]).imag
        print('time: {:.4f}, ehrenfest: {}, d1b: {}, tr: {}'.format(
               i*step, abs(error), imag, E[i].imag))
    return mu, E

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
#            ir1, ir2 = direct_spin1.trans_rdm12s(self.imag, self.real, norb, nelec)
#            check  = np.linalg.norm(ir1[0]-ri1[0].T)
#            check += np.linalg.norm(ir1[1]-ri1[1].T)
#            check += np.linalg.norm(ir2[0]-ri2[0].transpose(1,0,3,2))
#            check += np.linalg.norm(ir2[1]-ri2[1].transpose(1,0,3,2))
#            check += np.linalg.norm(ir2[2]-ri2[2].transpose(1,0,3,2))
#            print('check transpose: {}'.format(check))
#            check  = np.linalg.norm(rr2[0]-rr2[2])
#            check += np.linalg.norm(rr2[1]-rr2[1].transpose(2,3,0,1))
#            check += np.linalg.norm(ii2[0]-ii2[2])
#            check += np.linalg.norm(ii2[1]-ii2[1].transpose(2,3,0,1))
#            check += np.linalg.norm(ri2[0]-ri2[3])
#            check += np.linalg.norm(ri2[1]-ri2[2].transpose(2,3,0,1))
#            print(check)
            d2aa = rr2[0] + ii2[0] + 1j * ri2[0] - 1j * ri2[0].transpose(1,0,3,2)
            d2ab = rr2[1] + ii2[1] + 1j * ri2[1] - 1j * ri2[2].transpose(3,2,1,0)
            d2bb = rr2[2] + ii2[2] + 1j * ri2[3] - 1j * ri2[3].transpose(1,0,3,2)
            # transpose so that 2pdm[q,p,s,r] = \langle p^\dagger r^\dagger s q\rangle
            d2aa = d2aa.transpose(1,0,3,2) 
            d2ab = d2ab.transpose(1,0,3,2)
            d2bb = d2bb.transpose(1,0,3,2)
        else: 
            rr1 = direct_spin1.make_rdm1s(self.real, norb, nelec)
            ii1 = direct_spin1.make_rdm1s(self.imag, norb, nelec)
            ri1 = direct_spin1.trans_rdm1s(self.real, self.imag, norb, nelec)
            d2aa = d2ab = d2bb = None
        d1a = rr1[0] + ii1[0] + 1j * ri1[0] - 1j * ri1[0].T
        d1b = rr1[1] + ii1[1] + 1j * ri1[1] - 1j * ri1[1].T
#        if compute_d2:
#            check  = np.linalg.norm(d1a-d1a.T.conj())
#            check += np.linalg.norm(d1b-d1b.T.conj())
#            check += np.linalg.norm(d2aa-d2aa.transpose(1,0,3,2).conj())
#            check += np.linalg.norm(d2ab-d2ab.transpose(1,0,3,2).conj())
#            check += np.linalg.norm(d2bb-d2bb.transpose(1,0,3,2).conj())
#            print('check hermitian: {}'.format(check))
        return (d1a, d1b), (d2aa, d2ab, d2bb)

    def update_ci(self, real, imag, h1e, eri, E0, nelec):
        dr = update_ci(imag, h1e, eri, E0, nelec)
        di = update_ci(real, h1e, eri, E0, nelec)
        return - dr, di

    def update_RK4(self, r, i, h1e, eri, E0, nelec, step, RK4=True):
        dr1, di1 = self.update_ci(r, i, h1e, eri, E0, nelec)
        if not RK4:
            return dr1, di1
        else: 
            dr2, di2 = self.update_ci(r+dr1*step*0.5, i+di1*step*0.5, 
                                      h1e, eri, E0, nelec)
            dr3, di3 = self.update_ci(r+dr2*step*0.5, i+di2*step*0.5, 
                                      h1e, eri, E0, nelec)
            dr4, di4 = self.update_ci(r+dr3*step, i+di3*step, h1e, eri, E0, nelec)
            dr = (dr1+2.0*dr2+2.0*dr3+dr4)/6.0
            di = (di1+2.0*di2+2.0*di3+di4)/6.0
            return dr, di
      
    def compute_RHS(self, d1, d2, h1e, eri):
        d2aa, d2ab, d2bb = d2
        d2aa = d2aa.transpose(0,2,1,3)
        d2ab = d2ab.transpose(0,2,1,3)
        d2bb = d2bb.transpose(0,2,1,3)
        eri_ab = eri.transpose(0,2,1,3)
        eri_aa = eri_ab - eri_ab.transpose(0,1,3,2)

#        check  = np.linalg.norm(d2aa-d2bb)
#        check += np.linalg.norm(d2ab-d2ab.transpose(1,0,3,2))
#        print(check)
        def compute_imd(da, daa, dab, h, eriaa, eriab):
            C  = einsum('vp,pu->uv',da,h)
            C -= einsum('qu,vq->uv',da,h)
            C += 0.5 * einsum('pqus,vspq->uv',eriaa,daa)
            C +=       einsum('pQuS,vSpQ->uv',eriab,dab)
            C -= 0.5 * einsum('vqrs,rsuq->uv',eriaa,daa)
            C -=       einsum('vQrS,rSuQ->uv',eriab,dab)
            return C
        Ca = compute_imd(d1[0], d2aa, d2ab, h1e, eri_aa, eri_ab)
        Cb = compute_imd(d1[1], d2bb, d2ab.transpose(1,0,3,2), h1e, eri_aa, eri_ab.transpose(1,0,3,2))
#        print(np.linalg.norm(Ca-Cb))
        return 1j*Ca.T, 1j*Cb.T
    

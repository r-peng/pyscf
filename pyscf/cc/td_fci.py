from pyscf import lib, fci
from pyscf.fci import direct_spin1, cistring
import numpy as np
import math
einsum = lib.einsum

def update_ci(c, eris, time=None):
    eris.full_h(time)
    h2e = direct_spin1.absorb_h1e(eris.h, eris.eri, eris.nao, eris.nelec, 0.5)
    def _hop(c):
        return direct_spin1.contract_2e(h2e, c, eris.nao, eris.nelec)
    Hc = _hop(c)
    Hc -= eris.E0 * c
    h2e = None
    return - Hc

def update_RK(c, eris, step, RK=4):
    dc1 = update_ci(c, eris)
    if RK == 1:
        return dc1
    if RK == 4: 
        dc2 = update_ci(c+dc1*step*0.5, eris)
        dc3 = update_ci(c+dc2*step*0.5, eris)
        dc4 = update_ci(c+dc3*step    , eris)
        dc = (dc1+2.0*dc2+2.0*dc3+dc4)/6.0
        dc1 = dc2 = dc3 = dc4 = None
        return dc

def compute_energy(d1, d2, eris, time=None):
    eris.full_h(time)
    eri_ab = eris.eri.transpose(0,2,1,3).copy()
    eri_aa = eri_ab - eri_ab.transpose(0,1,3,2)
    e  = einsum('pq,qp',eris.h,d1[0])
    e += einsum('PQ,QP',eris.h,d1[1])
    e += 0.25 * einsum('pqrs,rspq',eri_aa,d2[0])
    e += 0.25 * einsum('PQRS,RSPQ',eri_aa,d2[2])
    e +=        einsum('pQrS,rSpQ',eri_ab,d2[1])
    eri_aa = eri_ab = None
    return e

def compute_rdm(c, nao, nelec):
    d1, d2 = direct_spin1.make_rdm12s(c, nao, nelec)
    d2aa = d2[0].transpose(1,3,0,2).copy() 
    d2ab = d2[1].transpose(1,3,0,2).copy()
    d2bb = d2[2].transpose(1,3,0,2).copy()
    d2 = None
    return d1, (d2aa, d2ab, d2bb)

def trace_err(d1, d2, eris):
    # err1 = d_{pp} - N
    # err2 = d_{prqr}/(N-1) - d_{pq}
    err1  = abs(np.trace(d1[0])-eris.nelec[0])
    err1 += abs(np.trace(d1[1])-eris.nelec[1])
    d2_  = einsum('prqr->pq',d2[0])
    d2_ += einsum('prqr->pq',d2[1])
    d2_ += einsum('prqr->pq',d2[2])
    d2_ += einsum('rprq->pq',d2[1])
    d2_ /= sum(eris.nelec) - 1
    err2 = np.linalg.norm(d2_-d1[0]-d1[1])
    d2_ = None
    return err1, err2

def kernel_it(mf, step=0.01, maxiter=1000, thresh=1e-6, RK=4):
    eris = ERIs(mf)
    eris.full_h() 

    N = cistring.num_strings(eris.nao, eris.nelec[0])
    c = np.zeros((N,N))
    c[0,0] = 1.0
    E_old = mf.energy_elec()[0]
    for i in range(maxiter):
        dc = update_RK(c, eris, step, RK=RK)
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

def kernel_rt_test(mf, c, w, f0, td, tf, step, RK=4, mo_coeff=None):
    eris = ERIs(mf, w, f0, td, mo_coeff)
    c = CI(c)
    d1, d2 = c.compute_rdm12(c.real, c.imag, eris.nao, eris.nelec)
    e = compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+mf.energy_nuc()))

    N = int((tf+step*0.1)/step)
    E = np.zeros(N+1, dtype=complex) 
    tr = np.zeros(2) # trace error
    ehr = np.zeros(2) # ehrenfest error = d<U^{-1}p+qU>/dt - i<U^{-1}[H,p+q]U>
    for i in range(N+1):
        time = i*step
        d1, d2 = c.compute_rdm12(c.real, c.imag, eris.nao, eris.nelec)
        E[i] = compute_energy(d1, d2, eris, time=None)
        dr, di = c.update_RK(c.real, c.imag, eris, time, step, RK=RK)
        # Ehrenfest err 
        LHS = compute_derivative(c, eris, time)
        RHS = compute_RHS(d1, d2, eris, time)
        ehr += np.array((np.linalg.norm(LHS[0]-RHS[0]),np.linalg.norm(LHS[1]-RHS[1])))
        tr += np.array(trace_err(d1, d2, eris)) 
        print('time: {:.4f}, EE(mH): {}'.format(time, (E[i]-E[0]).real*1e3))
        # update CI
        c.real += step * dr
        c.imag += step * di
        norm = np.linalg.norm(c.real+1j*c.imag)
        c.real /= norm 
        c.imag /= norm
    print('trace error: ',tr)
    print('Ehrenfest error: ', ehr)
    print('imaginary part of energy: ', np.linalg.norm(E.imag))
    return (E - E[0]).real

# not used for now
#def kernel_rt(mf, c, w, f0, td, tf, step, RK=4, mo_coeff=None):
#    eris = ERIs(mf, w, f0, td, mo_coeff)
#    eris.full_h()
#    c = CI(c)
#
#    N = int((tf+step*0.1)/step)
#    mus = np.zeros((N+1,3),dtype=complex)
#    Hmu = np.zeros((N+1,3),dtype=complex)
#    E = np.zeros(N+1,dtype=complex)
#
#    d1, d2 = c.compute_rdm(eris.nao, eris.nelec, compute_d2=True)
#    mus[0,:], _ = electric_dipole(eris, d1, d2=None, time=None) 
#    E[0] = compute_energy(d1, d2, eris, time=None)
#    print('FCI energy check : {}'.format(E[0].real+mf.energy_nuc()))
#    tr = compute_trace(d1, d2, eris)
#    err = np.zeros(2,dtype=complex)
#    for i in range(N+1):
#        time = i * step 
#        dr, di, LHS = c.update_RK(c.real, c.imag, eris, time, step, RK=RK)
#        # computing observables
#        d1, d2 = c.compute_rdm(eris.nao, eris.nelec, compute_d2=True)
#        mus[i,:] = electric_dipole(d1, eris)
#        RHS = compute_RHS(d1, d2, eris, time)
#        err += np.array(np.linalg.norm(LHS[0]-RHS[0]),np.linalg.norm(LHS[1]-RHS[1]))
#        E[i] = compute_energy(d1, d2, eris, time=None)
#        tr += compute_trace(d1, d2, eris)
#        print('time: {:.4f}, E(mH): {}, mu: {}'.format(
#               time, (E[i]-E[0]).real*1e3, (mus[i,:].real-eris.nucl_dip)*1e3))
#        # update CI
#        c.real += step * dr
#        c.imag += step * di
#        norm = np.linalg.norm(c.real+1j*c.imag)
#        c.real /= norm 
#        c.imag /= norm
#    print('check error: {}'.format(err))
#    print('check trace: {}'.format(tr))
#    print('check E imag: {}'.format(np.linalg.norm(E.imag)))
#    print('check mu imag: {}'.format(np.linalg.norm(mus.imag)))
#    return mus.real-eris.nucl_dip, (E - E[0]).real

class ERIs():
    def __init__(self, mf, w=None, f0=None, td=None, mo_coeff=None):
        mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        hao = mf.get_hcore()
        eri_ao = mf.mol.intor('int2e_sph')
        self.nao = mf.mol.nao_nr()
        self.nelec = mf.mol.nelec

        self.h0 = einsum('uv,up,vq->pq',hao,mo_coeff,mo_coeff)
        eri = einsum('uvxy,up,vr->prxy',eri_ao,mo_coeff,mo_coeff)
        self.eri = einsum('prxy,xq,ys->prqs',eri,mo_coeff,mo_coeff)

        # the number component 
        # time-indepedent case: reference energy
        # time-dependent case: nuclear component of dipole
        self.E0 = mf.energy_elec()[0] 
        if td is not None:
            mu_ao = mf.mol.intor('int1e_r')
            h1ao = einsum('xuv,x->uv',mu_ao,f0)
            self.h1 = einsum('uv,up,vq->pq',h1ao,mo_coeff,mo_coeff)
            self.mu = einsum('xuv,up,vq->xpq',mu_ao,mo_coeff,mo_coeff)
            charges = mf.mol.atom_charges()
            coords  = mf.mol.atom_coords()
            self.nucl_dip = einsum('i,ix->x', charges, coords)
            self.E0 = np.dot(self.nucl_dip,f0)
            self.w = w
            self.f0 = f0
            self.td = td
            mu_ao = h1ao = None
        mo_coeff = hao = eri_ao = eri = None

    def full_h(self, time=None):
        self.h = self.h0.copy()
        if time is not None: 
            if time < self.td:
                evlp = math.sin(math.pi*time/self.td)**2
                osc = math.cos(self.w*(time-self.td*0.5))
                self.h += self.h1 * evlp * osc

class CI():
    def __init__(self, c):
        self.real = c.copy()
        self.imag = np.zeros_like(c)

    def compute_rdm1(self, r, i, norb, nelec):
        rr1 = direct_spin1.make_rdm1s(self.real, norb, nelec)
        ii1 = direct_spin1.make_rdm1s(self.imag, norb, nelec)
        ri1 = direct_spin1.trans_rdm1s(self.real, self.imag, norb, nelec)
        d1a = rr1[0] + ii1[0] + 1j * ri1[0] - 1j * ri1[0].T
        d1b = rr1[1] + ii1[1] + 1j * ri1[1] - 1j * ri1[1].T
        rr1 = ii1 = ri1 = None
        return d1a, d1b

    def compute_rdm12(self, r, i, norb, nelec):
        # direct_spin returns: 
        # 1pdm[p,q] = :math:`\langle q^\dagger p\rangle
        # 2pdm[p,q,r,s] = \langle p^\dagger r^\dagger s q\rangle
        rr1, rr2 = direct_spin1.make_rdm12s(r, norb, nelec)
        ii1, ii2 = direct_spin1.make_rdm12s(i, norb, nelec)
        ri1, ri2 = direct_spin1.trans_rdm12s(r, i, norb, nelec)
        d1a = rr1[0] + ii1[0] + 1j * ri1[0] - 1j * ri1[0].T
        d1b = rr1[1] + ii1[1] + 1j * ri1[1] - 1j * ri1[1].T
        d2aa = rr2[0] + ii2[0] + 1j * ri2[0] - 1j * ri2[0].transpose(1,0,3,2)
        d2ab = rr2[1] + ii2[1] + 1j * ri2[1] - 1j * ri2[2].transpose(3,2,1,0)
        d2bb = rr2[2] + ii2[2] + 1j * ri2[3] - 1j * ri2[3].transpose(1,0,3,2)
        # transpose so that 2pdm[q,s,p,r] = \langle p^\dagger r^\dagger s q\rangle
        d2aa = d2aa.transpose(1,3,0,2) 
        d2ab = d2ab.transpose(1,3,0,2)
        d2bb = d2bb.transpose(1,3,0,2)
        rr1 = ii1 = ri1 = None
        rr2 = ii2 = ri2 = None
        return (d1a, d1b), (d2aa, d2ab, d2bb)

    def update_ci(self, real, imag, eris, time):
        dr = update_ci(imag, eris, time) * (-1.0)
        di = update_ci(real, eris, time)
        return dr, di

    def update_RK(self, r, i, eris, t, h, RK=4):
        dr1, di1 = self.update_ci(r, i, eris, t)
        if RK == 1:
            return dr1, di1
        if RK == 2:
            dr2, di2 = self.update_ci(r+dr1*h*0.5, i+di1*h*0.5, eris, t+h*0.5) 
            return dr2, di2
        if RK == 4: 
            dr2, di2 = self.update_ci(r+dr1*h*0.5, i+di1*h*0.5, eris, t+h*0.5) 
            dr3, di3 = self.update_ci(r+dr2*h*0.5, i+di2*h*0.5, eris, t+h*0.5)
            dr4, di4 = self.update_ci(r+dr3*h, i+di3*h, eris, t+h)
            dr = (dr1+2.0*dr2+2.0*dr3+dr4)/6.0
            di = (di1+2.0*di2+2.0*di3+di4)/6.0
            dr1 = dr2 = dr3 = dr4 = di1 = di2 = di3 = di4 = None
            return dr, di

def compute_derivative(c, eris, time):
# d/dt<psi|p+q|psi>/<psi|psi> =
# (d/dt<psi|p+q|psi>)/<psi|psi> + <psi|p+q|psi>d/dt<psi|psi>^-1
    dr, di = c.update_ci(c.real, c.imag, eris, time)
    norb = eris.nao
    nelec = eris.nelec 
    # term 1
    rr = direct_spin1.trans_rdm1s(dr, c.real, norb, nelec)
    ii = direct_spin1.trans_rdm1s(di, c.imag, norb, nelec)
    ri = direct_spin1.trans_rdm1s(dr, c.imag, norb, nelec)
    ir = direct_spin1.trans_rdm1s(di, c.real, norb, nelec)
    da = rr[0] + ii[0] + 1j * ri[0] - 1j * ir[0]
    db = rr[1] + ii[1] + 1j * ri[1] - 1j * ir[1]
    da += da.T.conj()
    db += db.T.conj()
    # term 2
    d1 = c.compute_rdm1(c.real, c.imag, norb, nelec)
    fac = einsum('IJ,IJ',dr-1j*di,c.real+1j*c.imag)
    fac += fac.conj()
    da -= fac*d1[0]
    db -= fac*d1[1]
    return da, db
 
def compute_RHS(d1, d2, eris, time=None):
    # F_{qp} = i<[H,p+q]>
    eris.full_h(time)
    eri_ab = eris.eri.transpose(0,2,1,3).copy()
    eri_aa = eri_ab - eri_ab.transpose(0,1,3,2)
    def compute_imd(da, daa, dab, eriaa, eriab):
        F  = einsum('vp,pu->uv',da,eris.h)
        F += 0.5 * einsum('pqus,vspq->uv',eriaa,daa)
        F +=       einsum('pQuS,vSpQ->uv',eriab,dab)
        return F - F.T.conj()
    Fa = compute_imd(d1[0], d2[0], d2[1], eri_aa, eri_ab)
    Fb = compute_imd(d1[1], d2[2], d2[1].transpose(1,0,3,2), 
                     eri_aa, eri_ab.transpose(1,0,3,2))
    eri_ab = eri_aa = None
    return 1j*Fa.T, 1j*Fb.T

def electric_dipole(d1, eris):
    mu  = einsum('xpq,qp->x',eris.mu,d1[0])
    mu += einsum('xpq,qp->x',eris.mu,d1[1])
    return mu

def compute_ci(t2, nfc=(0,0), nfv=(0,0)):
    # computes c = expT2|0>
    noa, nob, nva, nvb = t2[1].shape
    ne = noa + nfc[0], nob + nfc[1]
    nmo = ne[0] + nva + nfv[0], ne[1] + nvb + nfv[1]
    Na = cistring.num_strings(nmo[0],ne[0])
    Nb = cistring.num_strings(nmo[0],ne[1])
    ci = np.zeros((Na,Nb))
    ci[0,0] = 1.0

    Sa, Taa = ci_imt(nmo[0],nfc[0],t2[0])
    Sb, Tbb = ci_imt(nmo[1],nfc[1],t2[2])
    temp = einsum('KIia,ijab->KIjb',Sa,t2[1]) 
    def T(c):
        c_  = np.dot(Taa,c)
        c_ += np.dot(Tbb,c.T).T
        c_ += einsum('KIjb,LJjb,IJ->KL',temp,Sb,c)
        return c_
    out = ci.copy() 
    for n in range(1, sum(ne)+1):
        ci = T(ci)
        out += ci/math.factorial(n)
    ci = None
    return out/np.linalg.norm(out)

def ci_imt(nmo,nfc,t2):
    no, _, nv, _ = t2.shape
    ne = no + nfc
    N = cistring.num_strings(nmo,ne)
    S = np.zeros((N, N, no, nv))
    T = np.zeros((N, N))
    for I in range(N): 
        strI = cistring.addr2str(nmo,ne,I) 
        for i in range(no):
            des1 = i+nfc
            h1 = 1 << des1
            if strI & h1 != 0:
                for a in range(nv):
                    cre1 = a + ne
                    p1 = 1 << cre1
                    if strI & p1 == 0:
                        str1 = strI ^ h1 | p1
                        K = cistring.str2addr(nmo,ne,str1)
                        sgn1 = cistring.cre_des_sign(cre1,des1,strI)
                        S[K,I,i,a] += sgn1
                        for j in range(i):
                            des2 = j+nfc
                            h2 = 1 << des2
                            if strI & h2 != 0:
                                for b in range(a):
                                    cre2 = b + ne
                                    p2 = 1 << cre2
                                    if strI & p2 == 0:
                                        str2 = str1 ^ h2 | p2
                                        K = cistring.str2addr(nmo,ne,str2)
                                        sgn2 = cistring.cre_des_sign(cre2,des2,str1)
                                        T[K,I] += t2[i,j,a,b]*sgn1*sgn2
    return S, T 


from pyscf import lib, fci, cc
from pyscf.fci import direct_uhf, cistring
import numpy as np
import math
einsum = lib.einsum

def update_ci(c, eris, time=None):
    eris.full_h(time)
    h2e = direct_spin1.absorb_h1e(eris.h, eris.eri, eris.nao, eris.nelec, 0.5)
    def _hop(c):
        return direct_spin1.contract_2e(h2e, c, eris.nao, eris.nelec)
    Hc = _hop(c)
    if time is not None:
        Hc -= eris.E0 * c * td_occd_slow.fac(eris.w, eris.td, time)
    h2e = None
    return - Hc

def update_RK(c, eris, step, RK=4):
    dc1 = update_ci(c, eris)
    if RK == 1:
        return dc1
    if RK == 4: 
        c_ = c+dc1*step*0.5
        c_ /= np.linalg.norm(c_)
        dc2 = update_ci(c_, eris)

        c_ = c+dc2*step*0.5
        c_ /= np.linalg.norm(c_)
        dc3 = update_ci(c_, eris)

        c_ = c+dc3*step
        c_ /= np.linalg.norm(c_)
        dc4 = update_ci(c_, eris)

        dc = (dc1+2.0*dc2+2.0*dc3+dc4)/6.0
        dc1 = dc2 = dc3 = dc4 = c_ = None
        return dc

def compute_energy(d1, d2, eris, time=None):
    h1e_a, h1e_b = eris.h1e_r
    g2e_aa, g2e_ab, g2e_bb = eris.g2e_r
    h1e_a = np.array(h1e_a,dtype=complex)
    h1e_b = np.array(h1e_b,dtype=complex)
    g2e_aa = np.array(g2e_aa,dtype=complex)
    g2e_ab = np.array(g2e_ab,dtype=complex)
    g2e_bb = np.array(g2e_bb,dtype=complex)
    if not eris.realH: 
        h1e_a += 1j*eris.h1e_i[0]
        h1e_b += 1j*eris.h1e_i[1]
        g2e_aa += 1j*g2e.h1e_i[0]
        g2e_ab += 1j*g2e.h1e_i[1]
        g2e_bb += 1j*g2e.h1e_i[2]
    d1a, d1b = d1
    d2aa, d2ab, d2bb = d2
    # to physicts notation
    g2e_aa = g2e_aa.transpose(0,2,1,3)
    g2e_ab = g2e_ab.transpose(0,2,1,3)
    g2e_bb = g2e_bb.transpose(0,2,1,3)
    d2aa = d2aa.transpose(0,2,1,3)
    d2ab = d2ab.transpose(0,2,1,3)
    d2bb = d2bb.transpose(0,2,1,3)
    # antisymmetrize integral
    g2e_aa -= g2e_aa.transpose(1,0,2,3)
    g2e_bb -= g2e_bb.transpose(1,0,2,3)

    e  = einsum('pq,qp',h1e_a,d1a)
    e += einsum('PQ,QP',h1e_b,d1b)
    e += 0.25 * einsum('pqrs,rspq',g2e_aa,d2aa)
    e += 0.25 * einsum('PQRS,RSPQ',g2e_bb,d2bb)
    e +=        einsum('pQrS,rSpQ',g2e_ab,d2ab)
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

def kernel_rt(eris, fcivec, step, RK=4):
    c = CI(fcivec)
    d1, d2 = c.compute_rdm12(c.real, c.imag, eris.nao, eris.nelec)
    e = compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+mf.energy_nuc()))
#
    M = c.real.shape[0]
    cisolver = fci.FCI(mf.mol, eris.mo_coeff) 
    cisolver.nroots = M**2
    cisolver.kernel()
    eci = np.array(cisolver.eci) - mf.energy_nuc()
    v2 = einsum('NIJ,IJ->N',np.array(cisolver.ci),c.real+1j*c.imag)
    v3 = einsum('NIJ,IJ->N',np.array(cisolver.ci),c.real+1j*c.imag)
    h1 = np.zeros((cisolver.nroots,)*2)
    for m in range(cisolver.nroots):
        for n in range(cisolver.nroots):
            d1 = direct_spin1.trans_rdm1s(cisolver.ci[m],cisolver.ci[n],eris.nao,eris.nelec)
            h1[m,n] = np.dot(electric_dipole(d1, eris), f0)
    h1 -= np.eye(cisolver.nroots) * eris.E0
    def update2(c, t):
        phase = np.exp(-1j*t*eci)
        Hc = np.dot(h1,np.multiply(c,phase))*td_occd_slow.fac(w, td, t)
        return -1j*np.multiply(Hc,phase.conj())
    def update3(c, t):
        Hc  = np.dot(h1,c)*td_occd_slow.fac(w, td, t)
        Hc += np.multiply(c,eci)
        return -1j * Hc
    def _RK(c, t, h, RK, update):
        dc1 = update(c, t)
        if RK == 1:
            return dc1
        if RK == 4:
            c_ = c + dc1*h*0.5
            c_ /= np.linalg.norm(c_)
            dc2 = update(c_, t + h*0.5)
            c_ = c + dc2*h*0.5
            c_ /= np.linalg.norm(c_)
            dc3 = update(c_, t + h*0.5)
            c_ = c + dc3*h
            c_ /= np.linalg.norm(c_)
            dc4 = update(c_, t + h)
            return (dc1+2.0*dc2+2.0*dc3+dc4)/6.0
#
    N = int((tf+step*0.1)/step)
    E = np.zeros(N+1, dtype=complex) 
    E2 = np.zeros(N+1, dtype=complex)
    mu = np.zeros((N+1,3),dtype=complex)
    tr = np.zeros(2) # trace error
    ehr = np.zeros(2) # ehrenfest error = d<U^{-1}p+qU>/dt - i<U^{-1}[H,p+q]U>
    up2 = e2 = 0.0
    up3 = e3 = 0.0
    for i in range(N+1):
        time = i*step
        d1, d2 = c.compute_rdm12(c.real, c.imag, eris.nao, eris.nelec)
        E[i] = compute_energy(d1, d2, eris, time=None) # <H^0>
#        E[i] = compute_energy(d1, d2, eris, time) - eris.E0 * td_occd_slow.fac(w, td, time) # <H>
        mu[i,:] = electric_dipole(d1, eris) 
        dr, di = c.update_RK(c.real, c.imag, eris, time, step, RK)
        # Ehrenfest err 
        LHS = compute_derivative(c, eris, time)
        RHS = compute_RHS(d1, d2, eris, time)
        ehr += np.array((np.linalg.norm(LHS[0]-RHS[0]),np.linalg.norm(LHS[1]-RHS[1])))
        tr += np.array(trace_err(d1, d2, eris)) 
        # update CI
        c.real += step * dr
        c.imag += step * di
        norm = np.linalg.norm(c.real+1j*c.imag)
        c.real /= norm 
        c.imag /= norm
# scheme2
        phase = np.exp(-1j*time*eci)
        E2[i] = einsum('N,N,N',v2.conj(),v2,eci) # <H^0>
#        E2[i] += einsum('M,N,MN',np.multiply(v2,phase).conj(),np.multiply(v2,phase),h1) * td_occd_slow.fac(w, td, time) # <H^1>
        e2 += E2[i] - E[i]
        dv = _RK(v2, time, step, RK, update2)
        v2 += dv * step
        v2 /= np.linalg.norm(v2)
        v = einsum('N,NIJ->IJ',np.multiply(v2,phase),np.array(cisolver.ci))
        up2 += np.linalg.norm(c.real+1j*c.imag-v)
# scheme2
# scheme3
        e3 += einsum('N,N,N',v3.conj(),v3,eci) - E[i] # err(<H^0>)
#        e3 += einsum('N,N,N',v3.conj(),v3,eci) + einsum('M,N,MN',v3.conj(),v3,h1) * td_occd_slow.fac(w, td, time) - E[i] # err(<H>)
        dv = _RK(v3, time, step, RK, update3)
        v3 += dv * step
        v3 /= np.linalg.norm(v3)
        v = einsum('N,NIJ->IJ',v3,np.array(cisolver.ci))
        up3 += np.linalg.norm(c.real+1j*c.imag-v)
# scheme3
        print('time: {:.4f}, EE(mH): {}'.format(time, (E[i]-E[0]).real*1e3))
    print('update2 error, ci: {}, EE(mH): {}'.format(up2, abs(e2)*1e3))
    print('update3 error, ci: {}, EE(mH): {}'.format(up3, abs(e3)*1e3))
    print('trace error: ',tr)
    print('Ehrenfest error: ', ehr)
    print('imaginary part of energy: ', np.linalg.norm(E.imag))
    return (E - E[0]).real, (E2-E2[0]).real, (mu - eris.nucl_dip).real

class ERIs():
    def __init__(self, h1e, g2e, mo_coeff):
        ''' SIAM-like model Hamiltonian
            h1e: 1-elec Hamiltonian in site basis 
            g2e: 2-elec Hamiltonian in site basis
                 chemists notation (pr|qs)=<pq|rs>
            mo_coeff: moa, mob 
        '''
        moa, mob = mo_coeff
        
        h1e_a = einsum('uv,up,vq->pq',h1e,moa,moa)
        h1e_b = einsum('uv,up,vq->pq',h1e,mob,mob)
        g2e_aa = einsum('uvxy,up,vr->prxy',g2e,moa,moa)
        g2e_aa = einsum('prxy,xq,ys->prqs',g2e_aa,moa,moa)
        g2e_ab = einsum('uvxy,up,vr->prxy',g2e,moa,moa)
        g2e_ab = einsum('prxy,xq,ys->prqs',g2e_ab,mob,mob)
        g2e_bb = einsum('uvxy,up,vr->prxy',g2e,mob,mob)
        g2e_bb = einsum('prxy,xq,ys->prqs',g2e_bb,mob,mob)

        self.mo_coeff = mo_coeff
        self.h1e_r = h1e_a, h1e_b
        self.g2e_r = g2e_aa, g2e_ab, g2e_bb
        self.realH = True

class CI():
    def __init__(self, fcivec, norb, nelec):
        '''
           fcivec: ground state uhf fcivec
           norb: size of site basis
           nelec: nea, neb
        '''
        self.r = fcivec.copy()
        self.i = np.zeros_like(fcivec)
        self.norb = norb
        self.nelec = nelec

    def compute_rdm1(self):
        rr = direct_uhf.make_rdm1s(self.r, self.norb, self.nelec)
        ii = direct_uhf.make_rdm1s(self.i, self.norb, self.nelec)
        ri = direct_uhf.trans_rdm1s(self.r, self.i, self.norb, self.nelec)
        d1a = rr[0] + ii[0] + 1j*(ri[0]-ri[0].T)
        d1b = rr[1] + ii[1] + 1j*(ri[1]-ri[1].T)
        return d1a, d1b

    def compute_rdm12(self):
        # 1pdm[q,p] = :math:`\langle p^\dagger q\rangle
        # 2pdm[p,r,q,s] = \langle p^\dagger q^\dagger s r\rangle
        rr1, rr2 = direct_uhf.make_rdm12s(self.r, self.norb, self.nelec)
        ii1, ii2 = direct_uhf.make_rdm12s(self.i, self.norb, self.nelec)
        ri1, ri2 = direct_uhf.trans_rdm12s(self.r, self.i, self.norb, self.nelec)
        # make_rdm12s returns (d1a, d1b), (d2aa, d2ab, d2bb)
        # trans_rdm12s returns (d1a, d1b), (d2aa, d2ab, d2ba, d2bb)
        d1a = rr1[0] + ii1[0] + 1j*(ri1[0]-ri1[0].T)
        d1b = rr1[1] + ii1[1] + 1j*(ri1[1]-ri1[1].T)
        d2aa = rr2[0] + ii2[0] + 1j*(ri2[0]-ri2[0].transpose(1,0,3,2))
        d2ab = rr2[1] + ii2[1] + 1j*(ri2[1]-ri2[2].transpose(3,2,1,0))
        d2bb = rr2[2] + ii2[2] + 1j*(ri2[3]-ri2[3].transpose(1,0,3,2))
        # 2pdm[r,p,s,q] = \langle p^\dagger q^\dagger s r\rangle
        d2aa = d2aa.transpose(1,0,3,2) 
        d2ab = d2ab.transpose(1,0,3,2)
        d2bb = d2bb.transpose(1,0,3,2)
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
            r_ = r+dr1*h*0.5
            i_ = i+di1*h*0.5
            norm = np.linalg.norm(r_ + 1j*i_)
            r_ /= norm
            i_ /= norm
            dr2, di2 = self.update_ci(r_, i_, eris, t+h*0.5) 
            return dr2, di2
        if RK == 4: 
            r_ = r+dr1*h*0.5
            i_ = i+di1*h*0.5
            norm = np.linalg.norm(r_ + 1j*i_)
            r_ /= norm
            i_ /= norm
            dr2, di2 = self.update_ci(r_, i_, eris, t+h*0.5) 

            r_ = r+dr2*h*0.5
            i_ = i+di2*h*0.5
            norm = np.linalg.norm(r_ + 1j*i_)
            r_ /= norm
            i_ /= norm
            dr3, di3 = self.update_ci(r_, i_, eris, t+h*0.5)

            r_ = r+dr3*h
            i_ = i+di3*h
            norm = np.linalg.norm(r_ + 1j*i_)
            r_ /= norm
            i_ /= norm
            dr4, di4 = self.update_ci(r_, i_, eris, t+h)

            dr = (dr1+2.0*dr2+2.0*dr3+dr4)/6.0
            di = (di1+2.0*di2+2.0*di3+di4)/6.0
            dr1 = dr2 = dr3 = dr4 = di1 = di2 = di3 = di4 = r_ = i_ = None
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
    fac  = einsum('IJ,IJ',dr-1j*di,c.real+1j*c.imag)
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

if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        #['H', ( 0.,-0.5   ,-1.   )],
        #['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()

    norb = m.mo_energy[0].size
    nea = (mol.nelectron+1) // 2
    neb = (mol.nelectron-1) // 2
    nelec = (nea, neb)
    mo_a = m.mo_coeff[0]
    mo_b = m.mo_coeff[1]
    print(mo_a)
    print(mo_b)
    print(m.mo_energy[0])
    print(m.mo_energy[1])
    exit()
    cis = FCISolver(mol)
    h1e_a = reduce(numpy.dot, (mo_a.T, m.get_hcore(), mo_a))
    h1e_b = reduce(numpy.dot, (mo_b.T, m.get_hcore(), mo_b))
    g2e_aa = ao2mo.incore.general(m._eri, (mo_a,)*4, compact=False)
    g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
    g2e_ab = ao2mo.incore.general(m._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
    g2e_bb = ao2mo.incore.general(m._eri, (mo_b,)*4, compact=False)
    g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
    h1e = (h1e_a, h1e_b)
    eri = (g2e_aa, g2e_ab, g2e_bb)
    na = cistring.num_strings(norb, nea)
    nb = cistring.num_strings(norb, neb)
    numpy.random.seed(15)
    fcivec = numpy.random.random((na,nb))

    e = kernel(h1e, eri, norb, nelec)[0]
    print(e, e - -8.65159903476)

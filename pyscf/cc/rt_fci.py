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
    return -1j * Hc

def update_RK4(c, h1e, eri, E0, nelec, step, RK4=True):
    dc1 = update_ci(c, h1e, eri, E0, nelec, it=it)
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

def compute_RHS(d1, d2, h1e, eri):
    d2aa, d2ab, d2bb = d2
    d2aa = d2aa.transpose(0,2,1,3)
    d2ab = d2ab.transpose(0,2,1,3)
    d2bb = d2bb.transpose(0,2,1,3)
    eri_ab = eri.transpose(0,2,1,3)
    eri_aa = eri_ab - eri_ab.transpose(0,1,3,2)

    def compute_imd(d1a, d2aa, d2ab, h1e, eri_aa, eri_ab):
        C  = einsum('vp,pu->uv',d1a,h1e)
        C -= einsum('qu,vq->uv',d1a,h1e)
        C += 0.5 * einsum('pqus,vspq->uv',eri_aa,d2aa)
        C +=       einsum('pQuS,vSpQ->uv',eri_ab,d2ab)
        C -= 0.5 * einsum('vqrs,rsuq->uv',eri_aa,d2aa)
        C -=       einsum('vQrS,rSuQ->uv',eri_ab,d2ab)
        return C
    Ca = compute_imd(d1[0], d2aa, d2ab, h1e, eri_aa, eri_ab)
    Cb = compute_imd(d1[1], d2bb, d2ab.transpose(1,0,3,2), h1e, eri_aa, eri_ab.transpose(1,0,3,2))
    return 1j*Ca.T, 1j*Cb.T

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
    maxiter = int(tf/step)
    Rc = c.copy() # real part
    Ic = np.zeros_like(c) # imaginary part

    d1_old = direct_spin1.make_rdm1s(c, nao, nelec)
    tr  = abs(np.trace(d1_old[0])-nelec[0])
    print('check trace: {}'.format(tr))
    tr  = abs(np.trace(d1_old[1])-nelec[1])
    print('check trace: {}'.format(tr))
    print(d1_old[0].real)
    print(d1_old[0].imag)
    print(d1_old[1].real)
    print(d1_old[1].imag)
    exit()
    for i in range(maxiter):
        h1e = h.copy()
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
            osc = math.sin(w*i*step) 
            h1e += htd.copy() * osc * evlp
        dc = update_RK4(c, h1e, eri, E0, nelec, step, RK4=RK4, it=False)
        c += step * dc
        c /= np.linalg.norm(c)
        d1, d2 = direct_spin1.make_rdm12s(c, nao, nelec)
        dd1, d1_old = (d1[0]-d1_old[0], d1[1]-d1_old[1]), d1
        LHS = dd1[0]/step, dd1[1]/step
        RHS = compute_RHS(d1, d2, h1e, eri)
        error = np.linalg.norm(LHS[0]-RHS[0]), np.linalg.norm(LHS[1]-RHS[1])
        print('time: {:.4f}, d1a: {}, d1b: {}'.format(i*step, error[0], error[1]))
        if sum(error) > 1.0:
            print('diverging error!')
            break
        tr += abs(np.trace(d1_old[0])-nelec[0])
        tr += abs(np.trace(d1_old[1])-nelec[1])
    print('check trace: {}'.format(tr))

def kernel_rt(mf, t, l, U, w, f0, tp, tf, step, RK4=True, RK4_X=False):
    nao = mf.mol.nao_nr()
    mu_ao = mf.mol.intor('int1e_r')
    hao  = mu_ao[0,:,:] * f0[0]
    hao += mu_ao[1,:,:] * f0[1]
    hao += mu_ao[2,:,:] * f0[2]

    td = 2 * int(tp/step)
    maxiter = int(tf/step)
    no, _, nv, _ = l.shape
    mo0 = mf.mo_coeff.copy()
    U = np.array(U, dtype=complex)
    X = np.zeros((no+nv,)*2,dtype=complex)
    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    eris = ERIs(mf)
    mux = np.zeros(maxiter+1,dtype=complex)  
    muy = np.zeros(maxiter+1,dtype=complex)  
    muz = np.zeros(maxiter+1,dtype=complex)  
    Hmux = np.zeros(maxiter,dtype=complex)  
    Hmuy = np.zeros(maxiter,dtype=complex)  
    Hmuz = np.zeros(maxiter,dtype=complex)  
    E = np.zeros(maxiter,dtype=complex)

    d1, d2 = compute_rdms(t, l)
    mux_mo = ao2mo(mu_ao[0,:,:], mo_coeff)
    muy_mo = ao2mo(mu_ao[1,:,:], mo_coeff)
    muz_mo = ao2mo(mu_ao[2,:,:], mo_coeff)
    mux[0] = einsum('pq,qp',mux_mo,d1)
    muy[0] = einsum('pq,qp',muy_mo,d1)
    muz[0] = einsum('pq,qp',muz_mo,d1)

    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
#            osc = math.cos(w*(i*step-tp)) 
            osc = math.sin(w*i*step) 
            eris.h += ao2mo(hao, mo_coeff) * evlp * osc
        mux_mo = ao2mo(mu_ao[0,:,:], mo_coeff)
        muy_mo = ao2mo(mu_ao[1,:,:], mo_coeff)
        muz_mo = ao2mo(mu_ao[2,:,:], mo_coeff)
        dt, dl, X, C, d1, d2 = update_RK4(t, l, X, eris, step, RK4=RK4, RK4_X=RK4_X)
        t += step * dt
        l += step * dl
        mux[i+1] = einsum('pq,qp',mux_mo,d1)
        muy[i+1] = einsum('pq,qp',muy_mo,d1)
        muz[i+1] = einsum('pq,qp',muz_mo,d1)
        e  = einsum('pq,qp',eris.h,d1) 
        e += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
        E[i] = e
        Hmux[i] = einsum('pq,qp',mux_mo,C)
        Hmuy[i] = einsum('pq,qp',muy_mo,C)
        Hmuz[i] = einsum('pq,qp',muz_mo,C)
        U = np.dot(U, scipy.linalg.expm(step*X))
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
        error  = (mux[i+1]-mux[i])/step - Hmux[i] 
        error += (muy[i+1]-muy[i])/step - Hmuy[i] 
        error += (muz[i+1]-muz[i])/step - Hmuz[i]
        imag  = mux[i+1].imag + muy[i+1].imag + muz[i+1].imag 
        print('time: {:.4f}, ehrenfest: {}, imag: {}, E.imag: {},'.format(i*step, abs(error), imag, E[i].imag))
#        print('mux: {}, muy: {}, muz: {}'.format(mux[i+1].real,muy[i+1].real,muz[i+1].real))
    return mux, muy, muz, E

class CI():
    def __init__(self, c):
        self.real = c.copy()
        self.imag = np.zeros_like(c)

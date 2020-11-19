import numpy as np
import scipy
from pyscf import lib, ao2mo
einsum = lib.einsum

def init_amps(eris, mo_coeff, no):
    noa, nob = no
    eris.ao2mo(mo_coeff)
    fa, fb = eris.h[0].copy(), eris.h[1].copy()
    fa += einsum('piqi->pq',eris.eri[0][:,:noa,:,:noa])
    fa += einsum('pIqI->pq',eris.eri[1][:,:nob,:,:nob])
    fb += einsum('PIQI->PQ',eris.eri[2][:,:nob,:,:nob])
    fb += einsum('iPiQ->PQ',eris.eri[1][:noa,:,:noa,:])
    eoa = np.diag(fa[:noa,:noa])
    eva = np.diag(fa[noa:,noa:])
    eob = np.diag(fb[:nob,:nob])
    evb = np.diag(fb[nob:,nob:])
    eia = lib.direct_sum('i-a->ia', eoa, eva)
    eIA = lib.direct_sum('I-A->IA', eob, evb)
    eabij = lib.direct_sum('ia+jb->abij', eia, eia)
    eaBiJ = lib.direct_sum('ia+JB->aBiJ', eia, eIA)
    eABIJ = lib.direct_sum('IA+JB->ABIJ', eIA, eIA)
    taa = eris.eri[0][noa:,noa:,:noa,:noa]/eabij
    tab = eris.eri[1][noa:,nob:,:noa,:nob]/eaBiJ
    tbb = eris.eri[2][nob:,nob:,:nob,:nob]/eABIJ
    laa = taa.transpose(2,3,0,1).copy()
    lab = tab.transpose(2,3,0,1).copy()
    lbb = tbb.transpose(2,3,0,1).copy()
    return (taa, tab, tbb), (laa, lab, lbb)

def kernel_it(mf, maxiter=1000, step=0.03, thresh=1e-8, RK4=True):
    it = True
    noa, nob = mf.mol.nelec
    eris = ERIs(mf)
    mo0 = mf.mo_coeff.copy()
    Ua = np.eye(mf.mo_coeff[0].shape[0])
    Ub = np.eye(mf.mo_coeff[1].shape[0])
    mo_coeff = np.dot(mo0,Ua), np.dot(mo0,Ub)
    (taa, tab, tbb), (laa, lab, lbb) = init_amps(eris, mo_coeff, mf.mol.nelec)
    d1, d2 = compute_rdms((taa, tab, tbb), (laa, lab, lbb))
    e = compute_energy(d1, d2, eris)

    converged = False
    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        if RK4:
            dt, dl = update_RK4((taa, tab, tbb), (laa, lab, lbb), eris, step, it=it)
        else:
            dt, dl = update_amps((taa, tab, tbb), (laa, lab, lbb), eris, it=it)
        d1, d2 = compute_rdms((taa, tab, tbb), (laa, lab, lbb))
        X, _ = compute_X(d1, d2, eris, mf.mol.nelec, it)
        taa += step * dt[0]
        tab += step * dt[1]
        tbb += step * dt[2]
        laa += step * dl[0]
        lab += step * dl[1]
        lbb += step * dl[2]
        e_new = compute_energy(d1, d2, eris)
        de, e = e_new - e, e_new
        dnormX  = np.linalg.norm(X[0]) + np.linalg.norm(X[1])
        dnormt  = np.linalg.norm(dt[0])
        dnormt += np.linalg.norm(dt[1])
        dnormt += np.linalg.norm(dt[2])
        dnorml  = np.linalg.norm(dl[0])
        dnorml += np.linalg.norm(dl[1])
        dnorml += np.linalg.norm(dl[2])
        print('iter: {}, dX: {}, dt: {}, dl: {}, de: {}, energy: {}'.format(
              i, dnormX, dnormt, dnorml, de, e))
        if dnormX+dnormt+dnorml < thresh:
            converged = True
            break
        Ua = np.dot(Ua, scipy.linalg.expm(step*X[0])) # U = U_{old,new}
        Ub = np.dot(Ub, scipy.linalg.expm(step*X[1])) # U = U_{old,new}
        mo_coeff = np.dot(mo0,Ua), np.dot(mo0,Ub)
    return (taa, tab, tbb), (laa, lab, lbb), (Ua, Ub), e 

class ERIs:
    def __init__(self, mf):
        self.hao = mf.get_hcore().astype(complex)
        self.eri_ao = mf.mol.intor('int2e_sph').astype(complex)

    def ao2mo(self, mo_coeff):
        moa, mob = mo_coeff
        nmoa, nmob = moa.shape[0], mob.shape[0]
    
        ha = einsum('uv,up,vq->pq',self.hao,moa.conj(),moa)
        hb = einsum('uv,up,vq->pq',self.hao,mob.conj(),mob)
        self.h = ha, hb
    
        eri_aa = einsum('uvxy,up,vr->prxy',self.eri_ao,moa.conj(),moa)
        eri_aa = einsum('prxy,xq,ys->prqs',eri_aa,     moa.conj(),moa)
        eri_aa = eri_aa.transpose(0,2,1,3)
        eri_aa = eri_aa - eri_aa.transpose(0,1,3,2)
        eri_bb = einsum('uvxy,up,vr->prxy',self.eri_ao,mob.conj(),mob)
        eri_bb = einsum('prxy,xq,ys->prqs',eri_bb,     mob.conj(),mob)
        eri_bb = eri_bb.transpose(0,2,1,3)
        eri_bb = eri_bb - eri_bb.transpose(0,1,3,2)
        eri_ab = einsum('uvxy,up,vr->prxy',self.eri_ao,moa.conj(),moa)
        eri_ab = einsum('prxy,xq,ys->prqs',eri_ab,     mob.conj(),mob)
        eri_ab = eri_ab.transpose(0,2,1,3)
        self.eri = eri_aa.copy(), eri_ab.copy(), eri_bb.copy()

    fa, fb = eris.h[0].copy(), eris.h[1].copy()
    fa += einsum('piqi->pq',eri_aa[:,:noa,:,:noa])
    fa += einsum('pIqI->pq',eri_ab[:,:nob,:,:nob])
    fb += einsum('PIQI->PQ',eri_bb[:,:nob,:,:nob])
    fb += einsum('iPiQ->PQ',eri_ab[:noa,:,:noa,:])


import numpy as np
import scipy
from pyscf import lib, ao2mo
einsum = lib.einsum

def sort1(tup):
    a, b = tup
    na0, na1 = a.shape
    nb0, nb1 = b.shape
    out = np.zeros((na0+nb0,na1+nb1),dtype=complex)
    out[ ::2, ::2] = a.copy()
    out[1::2,1::2] = b.copy()
    return out

def sort2(tup, anti):
    aa, ab, bb = tup
    na0, na1, na2, na3 = aa.shape
    nb0, nb1, nb2, nb3 = bb.shape
    out = np.zeros((na0+nb0,na1+nb1,na2+nb2,na3+nb3),dtype=complex)
    out[ ::2, ::2, ::2, ::2] = aa.copy() 
    out[1::2,1::2,1::2,1::2] = bb.copy() 
    out[ ::2,1::2, ::2,1::2] = ab.copy()
    out[1::2, ::2,1::2, ::2] = ab.transpose(1,0,3,2).copy()
    if anti:
        out[ ::2,1::2,1::2, ::2] = - ab.transpose(0,1,3,2).copy()
        out[1::2, ::2, ::2,1::2] = - ab.transpose(1,0,2,3).copy()
    return out

def update_t(t, eris):
    no = t.shape[3]
    f = eris.f.copy()
    eri = eris.eri.copy()

    Foo  = f[:no,:no].copy()
    Foo += 0.5 * einsum('klcd,cdjl->kj',eri[:no,:no,no:,no:],t)
    Fvv  = f[no:,no:].copy()
    Fvv -= 0.5 * einsum('klcd,bdkl->bc',eri[:no,:no,no:,no:],t)

    dt  = eri[no:,no:,:no,:no].copy()
    dt += einsum('bc,acij->abij',Fvv,t)
    dt += einsum('ac,cbij->abij',Fvv,t)
    dt -= einsum('kj,abik->abij',Foo,t)
    dt -= einsum('ki,abkj->abij',Foo,t)

    dt += 0.5 * einsum('klij,abkl->abij',eri[:no,:no,:no,:no],t)
    dt += 0.5 * einsum('abcd,cdij->abij',eri[no:,no:,no:,no:],t)
    tmp = 0.5 * einsum('klcd,cdij->klij',eri[:no,:no,no:,no:],t)
    dt += 0.5 * einsum('klij,abkl->abij',tmp,t)

    tmp  = eri[:no,no:,no:,:no].copy()
    tmp += 0.5 * einsum('klcd,bdjl->kbcj',eri[:no,:no,no:,no:],t)
    tmp  = einsum('kbcj,acik->abij',tmp,t)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dt += tmp.copy()
    return dt

def update_l(t, l, eris):
    no = t.shape[3]
    f = eris.f.copy()
    eri = eris.eri.copy()

    Foo  = f[:no,:no].copy()
    Foo += 0.5 * einsum('ilcd,cdkl->ik',eri[:no,:no,no:,no:],t)
    Fvv  = f[no:,no:].copy()
    Fvv -= 0.5 * einsum('klad,cdkl->ca',eri[:no,:no,no:,no:],t)
    
    dl  = eri[:no,:no,no:,no:].copy() 
    dl += einsum('ca,ijcb->ijab',Fvv,l)
    dl += einsum('cb,ijac->ijab',Fvv,l)
    dl -= einsum('ik,kjab->ijab',Foo,l)
    dl -= einsum('jk,ikab->ijab',Foo,l)

    tmp  = 0.5 * einsum('ilcd,cdkl->ik',l,t)
    dl -= einsum('ik,kjab->ijab',tmp,eri[:no,:no,no:,no:])
    dl -= einsum('jk,ikab->ijab',tmp,eri[:no,:no,no:,no:])
    tmp  = 0.5 * einsum('klad,cdkl->ca',l,t)
    dl -= einsum('ca,ijcb->ijab',tmp,eri[:no,:no,no:,no:])
    dl -= einsum('cb,ijac->ijab',tmp,eri[:no,:no,no:,no:])

    vvvv  = eri[no:,no:,no:,no:].copy()
    vvvv += 0.5 * einsum('klab,cdkl->cdab',eri[:no,:no,no:,no:],t)
    oooo  = eri[:no,:no,:no,:no].copy()
    oooo += 0.5 * einsum('ijcd,cdkl->ijkl',eri[:no,:no,no:,no:],t)

    dl += 0.5 * einsum('cdab,ijcd->ijab',vvvv,l)
    dl += 0.5 * einsum('ijkl,klab->ijab',oooo,l)

    ovvo  = eri[:no,no:,no:,:no].copy()
    ovvo += einsum('jlbd,cdkl->jcbk',eri[:no,:no,no:,no:],t)
    tmp  = einsum('jcbk,ikac->ijab',ovvo,l)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dl += tmp.copy()
    return dl

def compute_gamma1(t, l): # normal ordered, asymmetric
    dvv = 0.5 * einsum('ikac,bcik->ba',l,t)
    doo = - 0.5 * einsum('jkac,acik->ji',l,t)
    return doo, dvv

def compute_gamma2(t, l): # normal ordered, asymmetric
    doovv = l.copy()
    dovvo = einsum('jkbc,acik->jabi',l,t)
    dvvvv = 0.5 * einsum('ijab,cdij->cdab',l,t)
    doooo = 0.5 * einsum('klab,abij->klij',l,t)
    dvvoo = t.copy()
    tmp  = einsum('acik,klcd->alid',t,l)
    tmp  = einsum('alid,bdjl->abij',tmp,t)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo += tmp.copy()
    tmp  = einsum('adkl,klcd->ac',t,l)
    tmp  = einsum('ac,cbij->abij',tmp,t)
    tmp -= tmp.transpose(1,0,2,3)
    dvvoo -= 0.5 * tmp.copy()
    tmp  = einsum('cdil,klcd->ki',t,l)
    tmp  = einsum('ki,abkj->abij',tmp,t)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo -= 0.5 * tmp.copy()
    tmp  = einsum('cdij,klcd->klij',t,l)
    dvvoo += 0.25 * einsum('klij,abkl->abij',tmp,t)
    return doooo, doovv, dvvoo, dovvo, dvvvv

def compute_rdms(t, l, normal=False, symm=True):
    doo, dvv = compute_gamma1(t, l)
    doooo, doovv, dvvoo, dovvo, dvvvv = compute_gamma2(t, l)

    no, nv = doo.shape[0], dvv.shape[0]
    nmo = no + nv
    if not normal:
        doooo += einsum('ki,lj->klij',np.eye(no),doo)
        doooo += einsum('lj,ki->klij',np.eye(no),doo)
        doooo -= einsum('li,kj->klij',np.eye(no),doo)
        doooo -= einsum('kj,li->klij',np.eye(no),doo)
        doooo += einsum('ki,lj->klij',np.eye(no),np.eye(no))
        doooo -= einsum('li,kj->klij',np.eye(no),np.eye(no))
        dovvo -= einsum('ji,ab->jabi',np.eye(no),dvv)
        doo += np.eye(no)

    d1 = np.zeros((nmo,nmo),dtype=complex)
    d1[:no,:no] = doo.copy()
    d1[no:,no:] = dvv.copy()
    d2 = np.zeros((nmo,nmo,nmo,nmo),dtype=complex)
    d2[:no,:no,:no,:no] = doooo.copy()
    d2[:no,:no,no:,no:] = doovv.copy()
    d2[no:,no:,:no,:no] = dvvoo.copy()
    d2[:no,no:,no:,:no] = dovvo.copy()
    d2[no:,:no,:no,no:] = dovvo.transpose(1,0,3,2)
    d2[:no,no:,:no,no:] = - dovvo.transpose(0,1,3,2)
    d2[no:,:no,no:,:no] = - dovvo.transpose(1,0,2,3)
    d2[no:,no:,no:,no:] = dvvvv.copy()

    if symm:
        d1 = 0.5 * (d1 + d1.T.conj())
        d2 = 0.5 * (d2 + d2.transpose(2,3,0,1).conj())
    return d1, d2

def compute_kappa_intermediates(d1, d2, eris, no):
    nv = d1.shape[0] - no
    h = eris.h.copy()
    eri = eris.eri.copy()

    Cov  = einsum('ba,aj->jb',d1[no:,no:],h[no:,:no]) 
    Cov -= einsum('ij,bi->jb',d1[:no,:no],h[no:,:no])
    Cov += 0.5 * einsum('pqjs,bspq->jb',eri[:,:,:no,:],d2[no:,:,:,:])
    Cov -= 0.5 * einsum('bqrs,rsjq->jb',eri[no:,:,:,:],d2[:,:,:no,:])

    Aovvo  = einsum('ba,ij->jbai',np.eye(nv),d1[:no,:no])
    Aovvo -= einsum('ij,ba->jbai',np.eye(no),d1[no:,no:])
    return Aovvo, Cov

def compute_kappa(d1, d2, eris, no):
    Aovvo, Cov = compute_kappa_intermediates(d1, d2, eris, no)
    nv = d1.shape[0] - no
    Aovvo = Aovvo.reshape(no*nv,no*nv)
    Cov = Cov.reshape(no*nv)
    kappa = np.dot(np.linalg.inv(Aovvo),Cov)
    kappa = kappa.reshape(nv,no)
    kappa = np.block([[np.zeros((no,no)),-kappa.T.conj()],
                      [kappa, np.zeros((nv,nv))]])
    return kappa

def compute_energy(d1, d2, eris):
    e  = einsum('pq,qp',eris.h,d1)
    e += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
    return e.real

def kernel_it1(mf, maxiter=1000, step=0.03, thresh=1e-6):
    no = sum(mf.mol.nelec)

    def kernel_t(eris):
        eo = np.diag(eris.f[:no,:no])
        ev = np.diag(eris.f[no:,no:])
        eia = lib.direct_sum('i-a->ia', eo, ev)
        eabij = lib.direct_sum('ia+jb->abij', eia, eia)
        t = eris.eri[no:,no:,:no,:no]/eabij

        converged = False
        for i in range(maxiter):
            dt = update_t(t, eris)
            dnorm = np.linalg.norm(dt)
            t -= step * dt
#            print('iter: {}, dnorm: {}'.format(i, dnorm))
            if dnorm < thresh*100:
                converged = True
                break
        if not converged: 
            print('t amplitude not converged!')
        return t

    def kernel_l(eris, t):
        l = t.transpose(2,3,0,1).copy()

        converged = False
        for i in range(maxiter):
            dl = update_l(t, l, eris)
            dnorm = np.linalg.norm(dl)
            l -= step * dl
#            print('iter: {}, dnorm: {}'.format(i, dnorm))
            if dnorm < thresh*100:
                converged = True
                break
        if not converged: 
            print('l amplitude not converged!')
        return l

    def compute_energy(eris, t):
        f = eris.f
        eri = eris.eri

        e  = einsum('ii',f[:no,:no])
        e -= 0.5 * einsum('ijij',eri[:no,:no,:no,:no])
        e += 0.25 * einsum('ijab,abij',eri[:no,:no,no:,no:],t)
        return e.real


    mo_coeff = mf.mo_coeff.copy()
    e = mf.energy_elec()[0]
    eris = ERIs(mf)

    converged = False
    for i in range(maxiter):
        t = kernel_t(eris)
        l = kernel_l(eris, t)
        e_new = compute_energy(eris, t)
        de, e = e_new - e, e_new
        if abs(de) < thresh:
            converged = True
            break
        d1, d2 = compute_rdms(t, l)
        kappa = compute_kappa(d1, d2, eris, no)
        dnorm = np.linalg.norm(kappa)
        print('iter: {}, dnorm: {}, de: {}, energy: {}'.format(i, dnorm, de, e))
        U = scipy.linalg.expm(step*kappa[::2,::2]) # U = U_{old,new}
        mo_coeff = np.dot(mo_coeff, U)
        eris.ao2mo(mo_coeff)
    return t, l, mo_coeff, e 

def kernel_it2(mf, maxiter=1000, step=0.03, thresh=1e-8):
    no = sum(mf.mol.nelec)
    eris = ERIs(mf)
    mo_coeff = mf.mo_coeff.copy()
    eo = np.diag(eris.f[:no,:no])
    ev = np.diag(eris.f[no:,no:])
    eia = lib.direct_sum('i-a->ia', eo, ev)
    eabij = lib.direct_sum('ia+jb->abij', eia, eia)
    t = eris.eri[no:,no:,:no,:no]/eabij
    l = t.transpose(2,3,0,1).copy()
    d1, d2 = compute_rdms(t, l)
    e = compute_energy(d1, d2, eris)

    converged = False
    for i in range(maxiter):
        kappa = compute_kappa(d1, d2, eris, no)
        U = scipy.linalg.expm(step*kappa[::2,::2]) # U = U_{old,new}
        mo_coeff = np.dot(mo_coeff, U)
        eris.ao2mo(mo_coeff)
        dt = update_t(t, eris)
        dl = update_l(t, l, eris)
        t -= step * dt
        l -= step * dl
        d1, d2 = compute_rdms(t, l)
        e_new = compute_energy(d1, d2, eris)
        de, e = e_new - e, e_new
        dnormk = np.linalg.norm(kappa)
        dnormt = np.linalg.norm(dt)
        dnorml = np.linalg.norm(dl)
        print('iter: {}, dk: {}, dt: {}, dl: {}, de: {}, energy: {}'.format(
              i, dnormk, dnormt, dnorml, de, e))
        if dnormk < thresh:
            converged = True
            break
    return t, l, mo_coeff, e 

class ERIs:
    def __init__(self, mf):
        self.hao = mf.get_hcore()
        self.fao = mf.get_fock()
        self.eri_ao = mf.mol.intor('int2e_sph')
        self.ao2mo(mf.mo_coeff)

    def ao2mo(self, mo_coeff):
        moa, mob = mo_coeff
        nmoa, nmob = moa.shape[0], mob.shape[0]
    
        ha = einsum('uv,up,vq->pq',self.hao,moa.conj(),moa)
        hb = einsum('uv,up,vq->pq',self.hao,mob.conj(),mob)
        self.h = sort1((ha,hb))
    
        fa = einsum('uv,up,vq->pq',self.fao[0],moa.conj(),moa)
        fb = einsum('uv,up,vq->pq',self.fao[1],mob.conj(),mob)
        self.f = sort1((fa,fb))
    
        eri_aa = einsum('uvxy,up,vr->prxy',self.eri_ao,moa.conj(),moa)
        eri_aa = einsum('prxy,xq,ys->prqs',eri_aa,     moa.conj(),moa)
        eri_aa = eri_aa.transpose(0,2,1,3)
        eri_bb = einsum('uvxy,up,vr->prxy',self.eri_ao,mob.conj(),mob)
        eri_bb = einsum('prxy,xq,ys->prqs',eri_bb,     mob.conj(),mob)
        eri_bb = eri_bb.transpose(0,2,1,3)
        eri_ab = einsum('uvxy,up,vr->prxy',self.eri_ao,moa.conj(),moa)
        eri_ab = einsum('prxy,xq,ys->prqs',eri_ab,     mob.conj(),mob)
        eri_ab = eri_ab.transpose(0,2,1,3)
        eri = sort2((eri_aa, eri_ab, eri_bb), anti=False)
        self.eri = eri - eri.transpose(0,1,3,2)

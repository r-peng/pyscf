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
    Foo  = eris.foo.copy()
    Foo += 0.5 * einsum('klcd,cdjl->kj',eris.oovv,t)
    Fvv  = eris.fvv.copy()
    Fvv -= 0.5 * einsum('klcd,bdkl->bc',eris.oovv,t)

    dt  = eris.oovv.transpose(2,3,0,1).conj().copy()
    dt += einsum('bc,acij->abij',Fvv,t)
    dt += einsum('ac,cbij->abij',Fvv,t)
    dt -= einsum('kj,abik->abij',Foo,t)
    dt -= einsum('ki,abkj->abij',Foo,t)

    dt += 0.5 * einsum('klij,abkl->abij',eris.oooo,t)
    dt += 0.5 * einsum('abcd,cdij->abij',eris.vvvv,t)
    tmp = 0.5 * einsum('klcd,cdij->klij',eris.oovv,t)
    dt += 0.5 * einsum('klij,abkl->abij',tmp,t)

    tmp  = eris.ovvo.copy()
    tmp += 0.5 * einsum('klcd,bdjl->kbcj',eris.oovv,t)
    tmp  = einsum('kbcj,acik->abij',tmp,t)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dt += tmp.copy()
    return dt

def update_l(t, l, eris):
    Foo  = eris.foo.copy()
    Foo += 0.5 * einsum('ilcd,cdkl->ik',eris.oovv,t)
    Fvv  = eris.fvv.copy()
    Fvv -= 0.5 * einsum('klad,cdkl->ca',eris.oovv,t)
    
    dl  = eris.oovv.copy() 
    dl += einsum('ca,ijcb->ijab',Fvv,l)
    dl += einsum('cb,ijac->ijab',Fvv,l)
    dl -= einsum('ik,kjab->ijab',Foo,l)
    dl -= einsum('jk,ikab->ijab',Foo,l)

    tmp  = 0.5 * einsum('ilcd,cdkl->ik',l,t)
    dl -= einsum('ik,kjab->ijab',tmp,eris.oovv)
    dl -= einsum('jk,ikab->ijab',tmp,eris.oovv)
    tmp  = 0.5 * einsum('klad,cdkl->ca',l,t)
    dl -= einsum('ca,ijcb->ijab',tmp,eris.oovv)
    dl -= einsum('cb,ijac->ijab',tmp,eris.oovv)

    vvvv  = eris.vvvv.copy()
    vvvv += 0.5 * einsum('klab,cdkl->cdab',eris.oovv,t)
    oooo  = eris.oooo.copy()
    oooo += 0.5 * einsum('ijcd,cdkl->ijkl',eris.oovv,t)

    dl += 0.5 * einsum('cdab,ijcd->ijab',vvvv,l)
    dl += 0.5 * einsum('ijkl,klab->ijab',oooo,l)

    ovvo  = eris.ovvo.copy()
    ovvo += einsum('jlbd,cdkl->jcbk',eris.oovv,t)
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

def compute_kappa_intermediates(t, l, eris):
    doo, dvv = compute_gamma1(t, l)
    doooo, doovv, dvvoo, dovvo, dvvvv = compute_gamma2(t, l)

    no, nv = doo.shape[0], dvv.shape[0]
    Aovvo  = einsum('ba,ij->jbai',np.eye(nv),doo+np.eye(no))
    Aovvo -= einsum('ij,ba->jbai',np.eye(no),dvv)

    Cov  = einsum('ba,ja->jb',dvv,eris.fov.conj())
    Cov -= einsum('ij,ib->jb',doo,eris.fov.conj())
    Cov -= eris.fov.conj().copy()
    Cov += 0.5 * einsum('abik,kija->jb',dvvoo,eris.ooov)
    Cov += 0.5 * einsum('abcd,jadc->jb',dvvvv,eris.ovvv.conj())
    Cov += einsum('ibak,jika->jb',dovvo,eris.ooov.conj())
    Cov -= 0.5 * einsum('acjk,kbca->jb',dvvoo,eris.ovvv)
    Cov -= 0.5 * einsum('iljk,likb->jb',doooo,eris.ooov.conj())
    Cov -= einsum('kacj,kacb->jb',dovvo,eris.ovvv.conj())
    Cov -= einsum('ac,jabc->jb',dvv,eris.ovvv.conj())
    Cov -= einsum('lk,ljkb->jb',doo,eris.ooov.conj())

    Cvo  = einsum('ji,ib->bj',doo,eris.fov)
    Cvo -= einsum('ab,ja->bj',dvv,eris.fov)
    Cvo -= eris.fov.T
    Cvo += 0.5 * einsum('ijac,ibac->bj',doovv,eris.ovvv.conj())
    Cvo += 0.5 * einsum('ijkl,klib->bj',doooo,eris.ooov)
    Cvo += einsum('jaci,icab->bj',dovvo,eris.ovvv)
    Cvo -= 0.5 * einsum('kiba,kija->bj',doovv,eris.ooov.conj())
    Cvo -= 0.5 * einsum('dcba,jadc->bj',dvvvv,eris.ovvv)
    Cvo -= einsum('kabi,jika->bj',dovvo,eris.ooov)
    Cvo += einsum('ca,jabc->bj',dvv,eris.ovvv)
    Cvo += einsum('ki,ijkb->bj',doo,eris.ooov)
    return Aovvo, Cov, Cvo

def compute_energy(eris, t, l, mo_coeff):
    e  = einsum('ii',eris.foo)
    e -= 0.5 * einsum('ijij',eris.oooo)
    e += 0.25 * einsum('ijab,abij',eris.oovv,t)

#    eri = einsum('uvxy,up,vr->prxy',eris.eri_ao,mo_coeff,mo_coeff)
#    eri = einsum('prxy,xq,ys->prqs',eri,mo_coeff,mo_coeff)
#    eri = eri.transpose(0,2,1,3)
#    eri = sort2((eri, eri, eri), anti=False)
#    eri -= eri.transpose(0,1,3,2)

#    no, nv = eris.fov.shape
#    e2  = einsum('ii',eris.foo)
#    e2 -= 0.5 * einsum('ijij',eris.oooo)
##    d1, d2 = compute_rdms(t, l, normal=True, symm=False)
#    d1, d2 = compute_rdms(t, l, normal=True)
#    e2 += einsum('ij,ji',eris.foo,d1[:no,:no])
#    e2 += einsum('ab,ba',eris.fvv,d1[no:,no:])
#    e2 += 0.25 * einsum('pqrs,rspq',eri,d2)
#    print('e - e2: {}'.format(e.real-e2.real))
    return e.real

def kernel_it(mf, maxiter=1000, step=0.03, thresh=1e-8):
    def kernel_t(eris):
        eo = np.diag(eris.foo)
        ev = np.diag(eris.fvv)
        eia = lib.direct_sum('i-a->ia', eo, ev)
        eabij = lib.direct_sum('ia+jb->abij', eia, eia)
        t = eris.oovv.transpose(2,3,0,1).conj()/eabij

        converged = False
        for i in range(maxiter):
            dt = update_t(t,eris)
            dnorm = np.linalg.norm(dt)
            t -= step * dt
#            print('iter: {}, dnorm: {}'.format(i, dnorm))
            if dnorm < thresh:
                converged = True
                break
        if not converged: 
            print('t amplitude not converged!')
        return t

    def kernel_l(eris, t):
        l = t.transpose(2,3,0,1).copy()

        converged = False
        for i in range(maxiter):
            dl = update_l(t,l,eris)
            dnorm = np.linalg.norm(dl)
            l -= step * dl
#            print('iter: {}, dnorm: {}'.format(i, dnorm))
            if dnorm < thresh:
                converged = True
                break
        if not converged: 
            print('l amplitude not converged!')
        return l

    def update_orbital(eris, t, l, mo_coeff):
        Aovvo, Cov, Cvo = compute_kappa_intermediates(t, l, eris) 
#        Aovvo -= Aovvo.transpose(3,2,1,0)
        Aovvo += Aovvo.transpose(3,2,1,0)
        Cov -= Cvo.T
        no, nv = Cov.shape
        nmo = no + nv

        d1, d2 = compute_rdms(t, l)
        A  = einsum('ab,ij->jbai',np.eye(nv),d1[:no,:no])
        A -= einsum('ij,ab->jbai',np.eye(no),d1[no:,no:])
        print('check A: {}'.format(np.linalg.norm(2*A-Aovvo)))
        h = einsum('uv,up,vq->pq',eris.hao,mo_coeff,mo_coeff)
        h = sort1((h,h))
        eri = einsum('uvxy,up,vr->prxy',eris.eri_ao,mo_coeff,mo_coeff)
        eri = einsum('prxy,xq,ys->prqs',eri,mo_coeff,mo_coeff)
        eri = eri.transpose(0,2,1,3)
        eri = sort2((eri, eri, eri), anti=False)
        eri -= eri.transpose(0,1,3,2)
        C1  = einsum('vp,pu->uv',d1,h) 
        C1 -= einsum('vp,qu->uv',h,d1)
        C1 += 0.5 * einsum('pqus,vspq->uv',eri,d2)
        C1 -= 0.5 * einsum('vqrs,rsuq->uv',eri,d2)

        d1, d2 = compute_rdms(t, l, normal=True)
        C2  = einsum('ba,ja->jb',d1[no:,no:],eris.fov.conj())
        C2 -= einsum('ij,ib->jb',d1[:no,:no],eris.fov.conj())
        C2 -= eris.fov.conj()
        C2 += 0.5 * einsum('abik,kija->jb',d2[no:,no:,:no,:no],eris.ooov)
        C2 += 0.5 * einsum('abcd,jadc->jb',d2[no:,no:,no:,no:],eris.ovvv.conj())
        C2 += einsum('ibak,jika->jb',d2[:no,no:,no:,:no],eris.ooov.conj())
        C2 -= 0.5 * einsum('acjk,kbca->jb',d2[no:,no:,:no,:no],eris.ovvv)
        C2 -= 0.5 * einsum('iljk,likb->jb',d2[:no,:no,:no,:no],eris.ooov.conj())
        C2 -= einsum('kacj,kacb->jb',d2[:no,no:,no:,:no],eris.ovvv.conj())
        C2 -= einsum('ac,jabc->jb',d1[no:,no:],eris.ovvv.conj())
        C2 -= einsum('lk,ljkb->jb',d1[:no,:no],eris.ooov.conj())

        C3  = einsum('ba,ja->jb',d1[no:,no:],eris.fov.conj())
        C3 -= einsum('ij,ib->jb',d1[:no,:no],eris.fov.conj())
        C3 -= eris.fov.conj()
        C3 += 0.5 * einsum('bvpq,pqjv->jb',d2[no:,:,:,:],eri[:,:,:no,:])
        print('check C: {}'.format(np.linalg.norm(2*C[:no,no:]-Cov)))
        print('check C2: {}'.format(np.linalg.norm(2*C_-Cov)))
        exit()

        Aovvo = Aovvo.reshape(no*nv,no*nv)
        Cov = Cov.reshape(no*nv)
        kappa = np.dot(np.linalg.inv(Aovvo),Cov)
        dnorm = np.linalg.norm(kappa)
        kappa = kappa.reshape(nv,no)
#        kappa = step * kappa.reshape(nv,no)
        kappa = np.block([[np.zeros((no,no)),kappa.T],
                          [-kappa, np.zeros((nv,nv))]])
        U = scipy.linalg.expm(kappa[::2,::2]) # U = U_{old,new}
        mo_coeff = np.dot(mo_coeff, U)
        return mo_coeff, dnorm 

    eris = ERIs(mf)
    mo_coeff = mf.mo_coeff.copy()
    e = mf.energy_elec()[0]

    converged = False
    for i in range(maxiter):
        t = kernel_t(eris)
        l = kernel_l(eris, t)
        e_new = compute_energy(eris, t, l, mo_coeff)
        mo_coeff, dnorm = update_orbital(eris, t, l, mo_coeff)
        de, e = e_new - e, e_new
        print('iter: {}, dnorm: {}, de: {}, energy: {}'.format(i, dnorm, de, e))
        eris.update_hamiltonian(mo_coeff)
#        exit()
        if abs(de) < thresh:
            converged = True
            break
    return t, l, mo_coeff 

class ERIs:
    def __init__(self, mf):
        self.mf = mf
        self.hao = mf.get_hcore()
        self.fock_ao = mf.get_fock().astype(complex)
        self.eri_ao = mf.mol.intor('int2e_sph').astype(complex)

        mo_coeff = mf.mo_coeff
#        eri = mf.mol.intor('int2e_sph', aosym='s8')
#        eri = ao2mo.incore.full(eri, mo_coeff)
#        eri = ao2mo.restore(1, eri, mf.mol.nao_nr())
#        eri_ = einsum('uvxy,up,vr->prxy',self.eri_ao,mo_coeff,mo_coeff)
#        eri_ = einsum('prxy,xq,ys->prqs',eri_,mo_coeff,mo_coeff)
#        print('check integral tranformation: {}'.format(np.linalg.norm(eri-eri_)))
#        f0 = einsum('uv,up,vq->pq',self.fock_ao,mo_coeff,mo_coeff)
#        print('check fock: {}'.format(np.linalg.norm(f0-np.diag(mf.mo_energy))))
#        exit()

        self.update_hamiltonian(mf.mo_coeff)

#        e  = einsum('ii',self.foo)
#        e -= 0.5 * einsum('ijij',self.oooo)
#        h = einsum('uv,up,vq->pq',self.hao,mo_coeff,mo_coeff)
#        h = sort1((h, h))
#        no, nv = self.fov.shape
#        e  = einsum('ii',h[:no,:no])
#        e += 0.5 * einsum('ijij',self.oooo)
#        eri = einsum('uvxy,up,vr->prxy',self.eri_ao,mo_coeff,mo_coeff)
#        eri = einsum('prxy,xq,ys->prqs',eri,mo_coeff,mo_coeff)
#        eri = eri.transpose(0,2,1,3)
#        eri = sort2((eri, eri, eri), anti=False)
#        eri -= eri.transpose(0,1,3,2)
#        t = np.zeros((nv,nv,no,no),dtype=complex)
#        l = np.zeros((no,no,nv,nv),dtype=complex)
#        d1, d2 = compute_rdms(t, l)
#        e  = einsum('pq,qp',h,d1)
#        e += 0.25 * einsum('pqrs,rspq',eri,d2)
#        print('check HF energy: {}'.format(mf.energy_elec()[0]-e.real))
#        exit()

    def update_hamiltonian(self, mo_coeff):
        nmo = mo_coeff.shape[0]
        no = sum(self.mf.mol.nelec)

        f0 = einsum('uv,up,vq->pq',self.fock_ao,mo_coeff,mo_coeff)
        f0 = sort1((f0, f0))
        self.foo = f0[:no,:no].copy()
        self.fov = f0[:no,no:].copy()
        self.fvv = f0[no:,no:].copy()

        eri = einsum('uvxy,up,vr->prxy',self.eri_ao,mo_coeff,mo_coeff)
        eri = einsum('prxy,xq,ys->prqs',eri,mo_coeff,mo_coeff)
        eri = eri.transpose(0,2,1,3)
        eri = sort2((eri, eri, eri), anti=False)
        eri -= eri.transpose(0,1,3,2)
        self.oooo = eri[:no,:no,:no,:no].copy() 
        self.ooov = eri[:no,:no,:no,no:].copy()
        self.oovv = eri[:no,:no,no:,no:].copy() 
        self.ovvo = eri[:no,no:,no:,:no].copy() 
        self.ovvv = eri[:no,no:,no:,no:].copy()
        self.vvvv = eri[no:,no:,no:,no:].copy() 

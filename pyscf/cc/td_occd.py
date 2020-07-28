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
    d1, d2 = compute_rdms(t, l, normal=False, symm=True)
    no, nv = eris.fov.shape 

    Cov  = einsum('ba,ja->jb',d1[no:,no:],eris.hov.conj())
    Cov -= einsum('ij,ib->jb',d1[:no,:no],eris.hov.conj())
    Cov += 0.5 * einsum('abik,kija->jb',d2[no:,no:,:no,:no],eris.ooov)
    Cov += 0.5 * einsum('abcd,jadc->jb',d2[no:,no:,no:,no:],eris.ovvv.conj())
    Cov += einsum('ibak,jika->jb',d2[:no,no:,no:,:no],eris.ooov.conj())
    Cov -= 0.5 * einsum('acjk,kbca->jb',d2[no:,no:,:no,:no],eris.ovvv)
    Cov -= 0.5 * einsum('iljk,likb->jb',d2[:no,:no,:no,:no],eris.ooov.conj())
    Cov -= einsum('kacj,kacb->jb',d2[:no,no:,no:,:no],eris.ovvv.conj())

    Aovvo  = einsum('ba,ij->jbai',np.eye(nv),d1[:no,:no])
    Aovvo -= einsum('ij,ba->jbai',np.eye(no),d1[no:,no:])
    return Aovvo, Cov

def compute_kappa(t, l, eris):
    Aovvo, Cov = compute_kappa_intermediates(t, l, eris)
    no, nv = Cov.shape
    Aovvo = Aovvo.reshape(no*nv,no*nv)
    Cov = Cov.reshape(no*nv)
    kappa = np.dot(np.linalg.inv(Aovvo),Cov)
    kappa = kappa.reshape(nv,no)
    kappa = np.block([[np.zeros((no,no)),-kappa.T.conj()],
                      [kappa, np.zeros((nv,nv))]])
    return kappa

def compute_energy(eris, t, l):
    e  = einsum('ii',eris.foo)
    e -= 0.5 * einsum('ijij',eris.oooo)
    e += 0.25 * einsum('ijab,abij',eris.oovv,t)
    return e.real

def kernel_it1(mf, maxiter=1000, step=0.03, thresh=1e-8):
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

    eris = ERIs(mf)
    mo_coeff = mf.mo_coeff.copy()
    e = mf.energy_elec()[0]

    converged = False
    for i in range(maxiter):
        t = kernel_t(eris)
        l = kernel_l(eris, t)
        e_new = compute_energy(eris, t, l)
        de, e = e_new - e, e_new
        if abs(de) < thresh:
            converged = True
            break
        kappa = compute_kappa(t, l, eris)
        dnorm = np.linalg.norm(kappa)
        print('iter: {}, dnorm: {}, de: {}, energy: {}'.format(i, dnorm, de, e))
        U = scipy.linalg.expm(step*kappa[::2,::2]) # U = U_{old,new}
#        U = scipy.linalg.expm(kappa[::2,::2]) # U = U_{old,new}
        mo_coeff = np.dot(mo_coeff, U)
        eris.update_hamiltonian(mo_coeff)
    return t, l, mo_coeff, e 

def kernel_it2(mf, maxiter=1000, step=0.03, thresh=1e-8):

    eris = ERIs(mf)
    mo_coeff = mf.mo_coeff.copy()
    e = mf.energy_elec()[0]
    eo = np.diag(eris.foo)
    ev = np.diag(eris.fvv)
    eia = lib.direct_sum('i-a->ia', eo, ev)
    eabij = lib.direct_sum('ia+jb->abij', eia, eia)
    t = eris.oovv.transpose(2,3,0,1).conj()/eabij
    l = t.transpose(2,3,0,1).copy()

    converged = False
    for i in range(maxiter):
        dt = update_t(t,eris)
        dl = update_l(t,l,eris)
        e_new = compute_energy(eris, t, l)
        de, e = e_new - e, e_new
        if abs(de) < thresh:
            converged = True
            break
        kappa = compute_kappa(t, l, eris)
        dnormt = np.linalg.norm(dt)
        dnorml = np.linalg.norm(dl)
        dnormk = np.linalg.norm(kappa)
        print('iter: {}, dnorm: {}, de: {}, energy: {}'.format(i, dnorm, de, e))
        t -= step * dt
        l -= step * dl
        U = scipy.linalg.expm(step*kappa[::2,::2]) # U = U_{old,new}
        mo_coeff = np.dot(mo_coeff, U)
        eris.update_hamiltonian(mo_coeff)
#        t, l = rotate_amps(t, l, U)
    return t, l, mo_coeff 

class ERIs:
    def __init__(self, mf):
        self.mf = mf
        self.hao = mf.get_hcore()
        self.fock_ao = mf.get_fock().astype(complex)
        self.eri_ao = mf.mol.intor('int2e_sph').astype(complex)

        self.update_hamiltonian(mf.mo_coeff)

    def update_hamiltonian(self, mo_coeff):
        nmo = mo_coeff.shape[0]
        no = sum(self.mf.mol.nelec)

        h = einsum('uv,up,vq->pq',self.hao,mo_coeff.conj(),mo_coeff)
        h = sort1((h,h))
        self.hoo = h[:no,:no].copy()
        self.hov = h[:no,no:].copy()
        self.hvv = h[no:,no:].copy()

        f0 = einsum('uv,up,vq->pq',self.fock_ao,mo_coeff.conj(),mo_coeff)
        f0 = sort1((f0, f0))
        self.foo = f0[:no,:no].copy()
        self.fov = f0[:no,no:].copy()
        self.fvv = f0[no:,no:].copy()

        eri = einsum('uvxy,up,vr->prxy',self.eri_ao,mo_coeff.conj(),mo_coeff)
        eri = einsum('prxy,xq,ys->prqs',eri,mo_coeff.conj(),mo_coeff)
        eri = eri.transpose(0,2,1,3)
        eri = sort2((eri, eri, eri), anti=False)
        eri -= eri.transpose(0,1,3,2)
        self.oooo = eri[:no,:no,:no,:no].copy() 
        self.ooov = eri[:no,:no,:no,no:].copy()
        self.oovv = eri[:no,:no,no:,no:].copy() 
        self.ovvo = eri[:no,no:,no:,:no].copy() 
        self.ovvv = eri[:no,no:,no:,no:].copy()
        self.vvvv = eri[no:,no:,no:,no:].copy()


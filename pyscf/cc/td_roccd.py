import numpy as np
import scipy, math
from pyscf import lib, ao2mo
einsum = lib.einsum

def update_amps(t, l, eris, it):
    no = l.shape[0]
    eri = eris.eri.real.copy() if it else eris.eri.copy()
    t_ = t - t.transpose(0,1,3,2)
    l_ = l - l.transpose(0,1,3,2)
    eri_ = eri - eri.transpose(0,1,3,2)
    f  = eris.h.real.copy() if it else eris.h.copy()
    f += 2.0 * einsum('piqi->pq',eri[:,:no,:,:no])
    f -= einsum('piiq->pq',eri[:,:no,:no,:])

    Foo  = f[:no,:no].copy()
    Foo += 0.5 * einsum('klcd,cdjl->kj',eri_[:no,:no,no:,no:],t_)
    Foo +=       einsum('klcd,cdjl->kj',eri[:no,:no,no:,no:],t)
    Fvv  = f[no:,no:].copy()
    Fvv -= 0.5 * einsum('klcd,bdkl->bc',eri_[:no,:no,no:,no:],t_)
    Fvv -=       einsum('klcd,bdkl->bc',eri[:no,:no,no:,no:],t)

    dt  = eri[no:,no:,:no,:no].copy()
    dt += einsum('bc,acij->abij',Fvv,t)
    dt += einsum('ac,cbij->abij',Fvv,t)
    dt -= einsum('kj,abik->abij',Foo,t)
    dt -= einsum('ki,abkj->abij',Foo,t)

    dl  = eri[:no,:no,no:,no:].copy()
    dl += einsum('cb,ijac->ijab',Fvv,l)
    dl += einsum('ca,ijcb->ijab',Fvv,l)
    dl -= einsum('jk,ikab->ijab',Foo,l)
    dl -= einsum('ik,kjab->ijab',Foo,l)

    loooo  = eri[:no,:no,:no,:no].copy()
    loooo += einsum('klcd,cdij->klij',eri[:no,:no,no:,no:],t)
    lvvvv  = eri[no:,no:,no:,no:].copy()
    lvvvv += einsum('klab,cdkl->cdab',eri[:no,:no,no:,no:],t)

    dt += einsum('abcd,cdij->abij',eri[no:,no:,no:,no:],t)
    dt += einsum('klij,abkl->abij',loooo,t)

    dl += einsum('cdab,ijcd->ijab',lvvvv,l)
    dl += einsum('ijkl,klab->ijab',loooo,l)

    rovvo_  = einsum('klcd,bdjl->kbcj',eri_[:no,:no,no:,no:],t_)
    rovvo_ += einsum('klcd,bdjl->kbcj',eri[:no,:no,no:,no:],t)
    rovvo  = einsum('lkdc,bdjl->kbcj',eri[:no,:no,no:,no:],t_)
    rovvo += einsum('klcd,bdjl->kbcj',eri_[:no,:no,no:,no:],t)
    rovov  = einsum('kldc,dbil->kbic',eri[:no,:no,no:,no:],t)

    tmp  = einsum('kbcj,acik->abij',eri[:no,no:,no:,:no],t_)
    tmp += einsum('kbcj,acik->abij',eri_[:no,no:,no:,:no],t)
    tmp -= einsum('kbic,ackj->abij',eri[:no,no:,:no,no:],t)
    tmp += tmp.transpose(1,0,3,2)
    dt += tmp.copy()
    dt += einsum('kbcj,acik->abij',rovvo_,t)
    dt += einsum('kbcj,acik->abij',rovvo,t_)
    dt += einsum('kbic,ackj->abij',rovov,t)

    tmp  = einsum('jcbk,ikac->ijab',eri[:no,no:,no:,:no]+rovvo,l_)
    tmp += einsum('jcbk,ikac->ijab',eri_[:no,no:,no:,:no]+rovvo_,l)
    tmp -= einsum('ickb,kjac->ijab',eri[:no,no:,:no,no:]-rovov,l)
    tmp += tmp.transpose(1,0,3,2)
    dl += tmp.copy()

    Foo  = 0.5 * einsum('ilcd,cdkl->ik',l_,t_)
    Foo +=       einsum('ilcd,cdkl->ik',l,t)
    Fvv  = 0.5 * einsum('klad,cdkl->ca',l_,t_)
    Fvv +=       einsum('klad,cdkl->ca',l,t)

    dl -= einsum('ik,kjab->ijab',Foo,eri[:no,:no,no:,no:])
    dl -= einsum('jk,ikab->ijab',Foo,eri[:no,:no,no:,no:])
    dl -= einsum('ca,ijcb->ijab',Fvv,eri[:no,:no,no:,no:])
    dl -= einsum('cb,ijac->ijab',Fvv,eri[:no,:no,no:,no:])

    f = eri = eri_ = None
    t_ = l_ = tmp = None
    if it:
        return -dt.real, -dl.real
    else:
        return -1j*dt, 1j*dl

def compute_gamma(t, l): # normal ordered, asymmetric
    t_ = t - t.transpose(0,1,3,2)
    l_ = l - l.transpose(0,1,3,2)
    dvv  = 0.5 * einsum('ikac,bcik->ba',l_,t_)
    dvv +=       einsum('ikac,bcik->ba',l,t)
    doo  = 0.5 * einsum('jkac,acik->ji',l_,t_)
    doo +=       einsum('jkac,acik->ji',l,t)
    doo *= - 1.0

    dovvo  = einsum('jkbc,acik->jabi',l_,t)
    dovvo += einsum('jkbc,acik->jabi',l,t_)
    dovov  = - einsum('jkcb,caik->jaib',l,t)

    dvvvv = einsum('ijab,cdij->cdab',l,t)
    doooo = einsum('klab,abij->klij',l,t)

    dvvoo  = t.copy()
    dvvoo += einsum('ladi,bdjl->abij',dovvo,t)
    dvvoo -= einsum('laid,bdjl->abij',dovov,t)
    dvvoo += einsum('ladi,bdjl->abij',dovvo,t_)
    dvvoo -= einsum('lajd,bdli->abij',dovov,t)
    dvvoo -= einsum('ac,cbij->abij',dvv,t)
    dvvoo -= einsum('bc,acij->abij',dvv,t)
    dvvoo += einsum('ki,abkj->abij',doo,t)
    dvvoo += einsum('kj,abik->abij',doo,t)
    dvvoo += einsum('klij,abkl->abij',doooo,t) 
    t_ = l_ = None
    return doo, dvv, doooo, l, dvvoo, dovvo, dovov, dvvvv 

def compute_rdms(t, l, normal=False, symm=True):
    doo, dvv, doooo, doovv, dvvoo, dovvo, dovov, dvvvv = compute_gamma(t, l)

    no, nv, _, _ = dovvo.shape
    nmo = no + nv
    if not normal:
        doooo += einsum('ki,lj->klij',np.eye(no),doo)
        doooo += einsum('lj,ki->klij',np.eye(no),doo)
        doooo += einsum('ki,lj->klij',np.eye(no),np.eye(no))

        dovov += einsum('ji,ab->jaib',np.eye(no),dvv)
        doo += np.eye(no)

    d1 = np.zeros((nmo,nmo),dtype=complex)
    d1[:no,:no] = doo.copy()
    d1[no:,no:] = dvv.copy()
    d2 = np.zeros((nmo,nmo,nmo,nmo),dtype=complex)
    d2[:no,:no,:no,:no] = doooo.copy()
    d2[:no,:no,no:,no:] = doovv.copy()
    d2[no:,no:,:no,:no] = dvvoo.copy()
    d2[:no,no:,no:,:no] = dovvo.copy()
    d2[no:,:no,:no,no:] = dovvo.transpose(1,0,3,2).copy()
    d2[:no,no:,:no,no:] = dovov.copy()
    d2[no:,:no,no:,:no] = dovov.transpose(1,0,3,2).copy()
    d2[no:,no:,no:,no:] = dvvvv.copy()

    doo = dvv = doooo = doovv = dvvoo = dovvo = dovov = dvvvv = None
    if symm:
        d1 = 0.5 * (d1 + d1.T.conj())
        d2 = 0.5 * (d2 + d2.transpose(2,3,0,1).conj())
    return d1, d2 

def compute_X(d1, d2, eris, no, it):
    nmo = d1.shape[0]
    nv = nmo - no

    h = eris.h.real.copy() if it else eris.h.copy()
    eri = eris.eri.real.copy() if it else eris.eri.copy() 

    C  = einsum('vp,pu->uv',d1,h) 
    C += 2.0 * einsum('pqus,vspq->uv',eri,d2)
    C -=       einsum('qpus,vspq->uv',eri,d2)
    C -= C.T.conj()
    Cov = C[:no,no:].real.copy() if it else C[:no,no:].copy()

    A = einsum('vp,qu->uvpq',np.eye(nmo),d1)
    A -= A.transpose(1,0,3,2).conj()
    Aovvo = A[:no,no:,no:,:no].real.copy() if it else A[:no,no:,no:,:no].copy()

    Aovvo = Aovvo.reshape(no*nv,no*nv)
    Cov = Cov.reshape(no*nv)
    Xvo = np.dot(np.linalg.inv(Aovvo),Cov)
    Xvo = Xvo.reshape(nv,no)
    h = eri = A = Aovvo = Cov = None
    if it: 
        X = np.zeros((nmo,nmo))
        X[:no,no:] = - Xvo.T.real
        X[no:,:no] = Xvo.real
    else:
        X = np.zeros((nmo,nmo),dtype=complex)
        X[:no,no:] = Xvo.T.conj()
        X[no:,:no] = Xvo.copy()
        X *= 1j
        C *= 1j
    return X, C.T

def compute_energy(d1, d2, eris):
    e  = 2.0 * einsum('pq,qp',eris.h,d1)
    e += 2.0 * einsum('pqrs,rspq',eris.eri,d2)
    e -=       einsum('pqsr,rspq',eris.eri,d2)
    return e.real

def init_amps(eris, mo_coeff, no):
    eris.ao2mo(mo_coeff)
    f  = eris.h.real.copy()
    f += 2.0 * einsum('piqi->pq',eris.eri.real[:,:no,:,:no])
    f -= einsum('piiq->pq',eris.eri.real[:,:no,:no,:])
    eoa = np.diag(f[:no,:no])
    eva = np.diag(f[no:,no:])
    eia = lib.direct_sum('i-a->ia', eoa, eva)
    eabij = lib.direct_sum('ia+jb->abij', eia, eia)
    t = eris.eri.real[no:,no:,:no,:no]/eabij
    l = t.transpose(2,3,0,1).copy()
    return t.real, l.real

def update_RK(t, l, eris, step, RK, it):
    dt1, dl1 = update_amps(t, l, eris, it) 
    if RK == 1:
        return dt1, dl1
    if RK == 4: 
        dt2, dl2 = update_amps(t + dt1*step*0.5, l + dl1*step*0.5, eris, it) 
        dt3, dl3 = update_amps(t + dt2*step*0.5, l + dl2*step*0.5, eris, it) 
        dt4, dl4 = update_amps(t + dt3*step, l + dl3*step, eris, it) 
        dt = (dt1 + 2.0*dt2 + 2.0*dt3 + dt4)/6.0
        dl = (dl1 + 2.0*dl2 + 2.0*dl3 + dl4)/6.0
        dt1 = dt2 = dt3 = dt4 = dl1 = dl2 = dl3 = dl4 = None
        return dt, dl
 
def kernel_it(mf, maxiter=1000, step=0.03, thresh=1e-8, RK=1):
    it = True
    no = mf.mol.nelec[0]
    eris = ERIs(mf)
    U = np.eye(mf.mo_coeff.shape[0])
    t, l = init_amps(eris, mf.mo_coeff, no)
    d1, d2 = compute_rdms(t, l)
    e = compute_energy(d1, d2, eris)

    converged = False
    for i in range(maxiter):
        eris.ao2mo(np.dot(mf.mo_coeff,U))
        dt, dl = update_RK(t, l, eris, step, RK, it)
        d1, d2 = compute_rdms(t, l)
        X, _ = compute_X(d1, d2, eris, no, it)
        t += step * dt
        l += step * dl
        e_new = compute_energy(d1, d2, eris)
        de, e = e_new - e, e_new
        dnormX  = np.linalg.norm(X)
        dnormt  = np.linalg.norm(dt)
        dnorml  = np.linalg.norm(dl)
        print('iter: {}, X: {}, dt: {}, dl: {}, de: {}, energy: {}'.format(
              i, dnormX, dnormt, dnorml, de, e))
        if dnormX+dnormt+dnorml < thresh:
            converged = True
            break
        U = np.dot(U,scipy.linalg.expm(step*X)) # U = U_{old,new}
    return t.real, l.real, U.real, e.real 

def ao2mo(Aao, mo_coeff):
    return einsum('uv,up,vq->pq',Aao,mo_coeff.conj(),mo_coeff)

class ERIs:
    def __init__(self, mf):
        self.hao = mf.get_hcore()
        self.eri_ao = mf.mol.intor('int2e_sph')

    def ao2mo(self, mo_coeff):
        nmo = mo_coeff.shape[0]
        self.h = ao2mo(self.hao, mo_coeff) 
        eri = einsum('uvxy,up,vr->prxy',self.eri_ao,mo_coeff.conj(),mo_coeff)
        eri = einsum('prxy,xq,ys->prqs',eri,        mo_coeff.conj(),mo_coeff)
        self.eri = eri.transpose(0,2,1,3).copy()
        eri = None

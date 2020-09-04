import numpy as np
import scipy, math
from pyscf import lib, ao2mo
einsum = lib.einsum

def sort1(tup):
    # sort 2d-tensors into spin-orbital form
    a, b = tup
    na0, na1 = a.shape
    nb0, nb1 = b.shape
    out = np.zeros((na0+nb0,na1+nb1),dtype=complex)
    out[ ::2, ::2] = a.copy()
    out[1::2,1::2] = b.copy()
    return out

def sort2(tup, anti):
    # sort 4d-tensors into spin-orbital form
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

# not used, same as update_amps
def compute_res_t(t, eris):
    # res_t_{\mu} = <\mu|\bar{H}|0>
    no = t.shape[3]
    eri = eris.eri.copy()
    f  = eris.h.copy()
    f += einsum('piqi->pq',eri[:,:no,:,:no])

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

# not used, same as update_amps
def compute_res_l(t, l, eris):
    # res_l_{\mu} = <0|(1+L)[\bar{H},\mu+]|0>
    no = t.shape[3]
    eri = eris.eri.copy()
    f  = eris.h.copy()
    f += einsum('piqi->pq',eri[:,:no,:,:no])

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

def update_amps(t, l, eris, time=None):
    # res_t_{\mu} = <\mu|\bar{H}|0>
    # res_l_{\mu} = <0|(1+L)[\bar{H},\mu+]|0>
    eris.full_h(time)

    no = t.shape[3]
    eri = eris.eri.copy()
    f  = eris.h.copy()
    f += einsum('piqi->pq',eri[:,:no,:,:no])

    Foo  = f[:no,:no].copy()
    Foo += 0.5 * einsum('ilcd,cdkl->ik',eri[:no,:no,no:,no:],t)
    Fvv  = f[no:,no:].copy()
    Fvv -= 0.5 * einsum('klad,cdkl->ca',eri[:no,:no,no:,no:],t)

    dt  = eri[no:,no:,:no,:no].copy()
    dt += einsum('bc,acij->abij',Fvv,t)
    dt += einsum('ac,cbij->abij',Fvv,t)
    dt -= einsum('kj,abik->abij',Foo,t)
    dt -= einsum('ki,abkj->abij',Foo,t)
    
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

    dt += 0.5 * einsum('klij,abkl->abij',oooo,t)
    dt += 0.5 * einsum('abcd,cdij->abij',eri[no:,no:,no:,no:],t)

    dl += 0.5 * einsum('cdab,ijcd->ijab',vvvv,l)
    dl += 0.5 * einsum('ijkl,klab->ijab',oooo,l)

    ovvo = einsum('jlbd,cdkl->jcbk',eri[:no,:no,no:,no:],t)

    tmp  = einsum('kbcj,acik->abij',eri[:no,no:,no:,:no] + 0.5 * ovvo,t)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dt += tmp.copy()

    tmp  = einsum('jcbk,ikac->ijab',eri[:no,no:,no:,:no] + ovvo,l)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dl += tmp.copy()

    eri = f = None
    vvvv = oooo = ovvo = tmp = None
    return -1j*dt, 1j*dl

def compute_gamma10(t, l): 
    # gamma_{qp} = <0|(1+L)e^(-T){p+q}e^T|0> = <l|{p+q}|r>
    dvv = 0.5 * einsum('ikac,bcik->ba',l,t)
    doo = - 0.5 * einsum('jkac,acik->ji',l,t)
    return doo, dvv

def compute_gamma11(t, l, dt, dl): 
    # analytical 1st time-derivative of gamma_{qp}
    dvv  = 0.5 * einsum('ikac,bcik->ba',dl,t)
    dvv += 0.5 * einsum('ikac,bcik->ba',l,dt)
    doo  = - 0.5 * einsum('jkac,acik->ji',dl,t)
    doo += - 0.5 * einsum('jkac,acik->ji',l,dt)
    return doo, dvv

def compute_gamma20(t, l):
    # gamma_{rspq} = <0|(1+L)e^(-T){p+q+sr}e^T|0> = <l|{p+q+sr}|r>
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

def compute_gamma21(t, l, dt, dl):
    # analytical 1st time-derivative of gamma_{rspq}
    doovv  = dl.copy()
    dovvo  = einsum('jkbc,acik->jabi',dl,t)
    dovvo += einsum('jkbc,acik->jabi',l,dt)
    dvvvv  = 0.5 * einsum('ijab,cdij->cdab',dl,t)
    dvvvv += 0.5 * einsum('ijab,cdij->cdab',l,dt)
    doooo  = 0.5 * einsum('klab,abij->klij',dl,t)
    doooo += 0.5 * einsum('klab,abij->klij',l,dt)
    dvvoo = dt.copy()
    tl   = einsum('acik,klcd->alid',t,l)
    dtl  = einsum('acik,klcd->alid',dt,l)
    dtl += einsum('acik,klcd->alid',t,dl)
    tmp  = einsum('alid,bdjl->abij',tl,dt)
    tmp += einsum('alid,bdjl->abij',dtl,t)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo += tmp.copy()
    tl   = einsum('adkl,klcd->ac',t,l)
    dtl  = einsum('adkl,klcd->ac',dt,l)
    dtl += einsum('adkl,klcd->ac',t,dl)
    tmp  = einsum('ac,cbij->abij',tl,dt)
    tmp += einsum('ac,cbij->abij',dtl,t)
    tmp -= tmp.transpose(1,0,2,3)
    dvvoo -= 0.5 * tmp.copy()
    tl   = einsum('cdil,klcd->ki',t,l)
    dtl  = einsum('cdil,klcd->ki',dt,l)
    dtl += einsum('cdil,klcd->ki',t,dl)
    tmp  = einsum('ki,abkj->abij',tl,dt)
    tmp += einsum('ki,abkj->abij',dtl,t)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo -= 0.5 * tmp.copy()
    tl   = einsum('cdij,klcd->klij',t,l)
    dtl  = einsum('cdij,klcd->klij',dt,l)
    dtl += einsum('cdij,klcd->klij',t,dl)
    dvvoo += 0.25 * einsum('klij,abkl->abij',tl,dt)
    dvvoo += 0.25 * einsum('klij,abkl->abij',dtl,t)
    return doooo, doovv, dvvoo, dovvo, dvvvv

def compute_gamma22(t, l, dt, dl):
    # O(h^2) component of gamma_{rspq} from update t+dt*h, l+dl*h
    # h = time step size
    # not the same as analytical 2nd time-derivative
    dovvo = einsum('jkbc,acik->jabi',dl,dt)
    dvvvv = 0.5 * einsum('ijab,cdij->cdab',dl,dt)
    doooo = 0.5 * einsum('klab,abij->klij',dl,dt)
    dvvoo = np.zeros_like(t, dtype=complex)
    dtdl = einsum('acik,klcd->alid',dt,dl)
    dtl  = einsum('acik,klcd->alid',dt,l)
    dtl += einsum('acik,klcd->alid',t,dl)
    tmp  = einsum('alid,bdjl->abij',dtdl,t)
    tmp += einsum('alid,bdjl->abij',dtl,dt)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo += tmp.copy()
    dtdl = einsum('adkl,klcd->ac',dt,dl)
    dtl  = einsum('adkl,klcd->ac',dt,l)
    dtl += einsum('adkl,klcd->ac',t,dl)
    tmp  = einsum('ac,cbij->abij',dtdl,t)
    tmp += einsum('ac,cbij->abij',dtl,dt)
    tmp -= tmp.transpose(1,0,2,3)
    dvvoo -= 0.5 * tmp.copy()
    dtdl = einsum('cdil,klcd->ki',dt,dl)
    dtl  = einsum('cdil,klcd->ki',dt,l)
    dtl += einsum('cdil,klcd->ki',t,dl)
    tmp  = einsum('ki,abkj->abij',dtdl,t)
    tmp += einsum('ki,abkj->abij',dtl,dt)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo -= 0.5 * tmp.copy()
    dtdl = einsum('cdij,klcd->klij',dt,dl)
    dtl  = einsum('cdij,klcd->klij',dt,l)
    dtl += einsum('cdij,klcd->klij',t,dl)
    dvvoo += 0.25 * einsum('klij,abkl->abij',dtdl,t)
    dvvoo += 0.25 * einsum('klij,abkl->abij',dtl,dt)
    return doooo, dvvoo, dovvo, dvvvv

def compute_gamma23(dt, dl):
    # O(h^3) component of gamma_{rspq} from update t+dt*h, l+dl*h
    # h = time step size
    # not the same as analytical 3rd time-derivative
    dvvoo = np.zeros_like(dt, dtype=complex) 
    tmp  = einsum('acik,klcd->alid',dt,dl)
    tmp  = einsum('alid,bdjl->abij',tmp,dt)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo += tmp.copy()
    tmp  = einsum('adkl,klcd->ac',dt,dl)
    tmp  = einsum('ac,cbij->abij',tmp,dt)
    tmp -= tmp.transpose(1,0,2,3)
    dvvoo -= 0.5 * tmp.copy()
    tmp  = einsum('cdil,klcd->ki',dt,dl)
    tmp  = einsum('ki,abkj->abij',tmp,dt)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo -= 0.5 * tmp.copy()
    tmp  = einsum('cdij,klcd->klij',dt,dl)
    dvvoo += 0.25 * einsum('klij,abkl->abij',tmp,dt)
    return dvvoo

def compute_rdm1(t, l, dt=None, dl=None, order=0): 
    # d_{qp} = 0.5*(<l|p+q|r>+<r|p+q|l>) = <p+q>
    no, _, nv, _ = l.shape
    nmo = no + nv
    if order == 0:
        doo, dvv = compute_gamma10(t, l)
        doo += np.eye(no)
    if order == 1:
        doo, dvv = compute_gamma11(t, l, dt, dl)
    if order == 2:
        doo, dvv = compute_gamma10(dt, dl)

    d1 = np.zeros((nmo,nmo),dtype=complex)
    d1[:no,:no] += doo
    d1[no:,no:] += dvv
    d1 = 0.5 * (d1 + d1.T.conj())

    doo = dvv = None
    return d1

def compute_rdm12(t, l, dt=None, dl=None, order=0): 
    # d_{qp} = 0.5*(<l|p+q|r>+<r|p+q|l>) = <p+q>
    # d_{rspq} = 0.5*(<l|p+q+rs|r>+<r|p+q+sr|l>) = <p+q+sr>
    no, _, nv, _ = l.shape
    nmo = no + nv
    if order == 0:
        doo, dvv = compute_gamma10(t, l)
        doooo, doovv, dvvoo, dovvo, dvvvv = compute_gamma20(t, l)
        doooo += einsum('ki,lj->klij',np.eye(no),doo)
        doooo += einsum('lj,ki->klij',np.eye(no),doo)
        doooo -= einsum('li,kj->klij',np.eye(no),doo)
        doooo -= einsum('kj,li->klij',np.eye(no),doo)
        doooo += einsum('ki,lj->klij',np.eye(no),np.eye(no))
        doooo -= einsum('li,kj->klij',np.eye(no),np.eye(no))
        dovvo -= einsum('ji,ab->jabi',np.eye(no),dvv)
        doo += np.eye(no)
    if order == 1:
        doo, dvv = compute_gamma11(t, l, dt, dl)
        doooo, doovv, dvvoo, dovvo, dvvvv = compute_gamma21(t, l, dt, dl)
        doooo += einsum('ki,lj->klij',np.eye(no),doo)
        doooo += einsum('lj,ki->klij',np.eye(no),doo)
        doooo -= einsum('li,kj->klij',np.eye(no),doo)
        doooo -= einsum('kj,li->klij',np.eye(no),doo)
        dovvo -= einsum('ji,ab->jabi',np.eye(no),dvv)
    if order == 2: 
        doo, dvv = compute_gamma10(dt, dl)
        doooo, dvvoo, dovvo, dvvvv = compute_gamma22(t, l, dt, dl)
        doooo += einsum('ki,lj->klij',np.eye(no),doo)
        doooo += einsum('lj,ki->klij',np.eye(no),doo)
        doooo -= einsum('li,kj->klij',np.eye(no),doo)
        doooo -= einsum('kj,li->klij',np.eye(no),doo)
        dovvo -= einsum('ji,ab->jabi',np.eye(no),dvv)
        doovv = 0.0
    if order == 3: 
        dvvoo = compute_gamma23(dt, dl)
        doo = dvv = doooo = doovv = dvvvv = 0.0
        dovvo = np.zeros((no,nv,nv,no))

    d1 = np.zeros((nmo,nmo),dtype=complex)
    d1[:no,:no] += doo
    d1[no:,no:] += dvv
    d2 = np.zeros((nmo,nmo,nmo,nmo),dtype=complex)
    d2[:no,:no,:no,:no] += doooo
    d2[:no,:no,no:,no:] += doovv
    d2[no:,no:,:no,:no] += dvvoo
    d2[:no,no:,no:,:no] += dovvo
    d2[no:,:no,:no,no:] += dovvo.transpose(1,0,3,2)
    d2[:no,no:,:no,no:] -= dovvo.transpose(0,1,3,2)
    d2[no:,:no,no:,:no] -= dovvo.transpose(1,0,2,3)
    d2[no:,no:,no:,no:] += dvvvv
    d1 = 0.5 * (d1 + d1.T.conj())
    d2 = 0.5 * (d2 + d2.transpose(2,3,0,1).conj())

    doo = dvv = doooo = doovv = dvvoo = dovvo = dovov = dvvvv = None
    return d1, d2

def compute_X(d1, d2, eris, time, no):
    # A_{uv,pq}(-iX)_{pq} = F_{uv}
    # A_{uv,pq} = <[u+v,p+q]>
    # F_{uv} = <[U^{-1}HU,u+v]>
    eris.full_h(time)

    nmo = d1.shape[0]
    nv = nmo - no
    A  = einsum('vp,qu->uvpq',np.eye(nmo),d1)
    A -= A.transpose(1,0,3,2).conj()
    Aovvo = A[:no,no:,no:,:no].copy()

    F  = einsum('vp,pu->uv',d1,eris.h)
    F += 0.5 * einsum('pqus,vspq->uv',eris.eri,d2)
    F -= F.T.conj()
    Fov = F[:no,no:].copy()

    Aovvo = Aovvo.reshape(no*nv,nv*no)
    Fov = Fov.reshape(no*nv)
    
    Xvo = np.dot(np.linalg.inv(Aovvo),Fov)
    Xvo = Xvo.reshape(nv,no)
    
    X = np.zeros((nmo,nmo),dtype=complex)
    X[:no,no:] = Xvo.T.conj()
    X[no:,:no] = Xvo.copy()
    return 1j*X, 1j*F.T

def compute_der1(t, l, C, X, d1, eris, time):
    # analytical 1st time derivative of <U^{-1}p+qU>

    # amplitude component
    dt, dl = update_amps(t, l, eris, time)
    d11 = compute_rdm1(t, l, dt, dl, order=1)
    d11 = rotate1(d11, C.T.conj())

    # orbital component
    dC = - np.dot(X,C)
    tmp  = einsum('sr,rp,sq->qp',d1,dC,C.conj())
    tmp += einsum('sr,rp,sq->qp',d1,C,dC.conj())
    return d11 + tmp

def energy_err(t, l, X, d1, d2, eris, time):
    # analytical nonconservation error from derivative of wavefunction
    # err = d<U^{-1}HU>/dt - <U^{-1}\dot{H}U>
    #     = <\dot{l}|U^{-1}HU|r> + <l|U^{-1}HU|\dot{r}> # amplitude component 
    #     + <l|\dot{U^{-1}}HU|r> + <l|U^{-1}H\dot{U}|r> # orbital component
    # each component should identically vanish

    # amplitude component
    dt, dl = update_amps(t, l, eris, time)
    d11, d21 = compute_rdm12(t, l, dt, dl, order=1)
    err_amp = compute_energy(d11, d21, eris, time) 
    # orbital component
    tmp1  = einsum('ur,rv->uv',eris.h,X)
    tmp1 -= einsum('ur,rv->uv',X,eris.h)
    tmp2  = 0.5 * einsum('uvxw,wy->uvxy',eris.eri,X)
    tmp2 -= 0.5 * einsum('uwxy,vw->uvxy',eris.eri,X)
    err_orb  = einsum('uv,vu',tmp1,d1)
    err_orb += einsum('uvxy,xyuv',tmp2,d2)
    return abs(err_amp), abs(err_orb)

def compute_energy(d1, d2, eris, time=None):
    eris.full_h(time)
    e  = einsum('pq,qp',eris.h,d1)
    e += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
    return e.real

def update_RK(t, l, eris, time, h, RK):
    # Runge-Kutta 
    dt1, dl1 = update_amps(t, l, eris, time)
    if RK == 1:
        return dt1, dl1
    if RK == 2:
        dt2, dl2 = update_amps(t + dt1*h*0.5, l + dl1*h*0.5, eris, time+h*0.5) 
        return dt2, dl2
    if RK == 4: 
        dt2, dl2 = update_amps(t + dt1*h*0.5, l + dl1*h*0.5, eris, time+h*0.5) 
        dt3, dl3 = update_amps(t + dt2*h*0.5, l + dl2*h*0.5, eris, time+h*0.5) 
        dt4, dl4 = update_amps(t + dt3*h, l + dl3*h, eris, time+h) 
        dt = (dt1 + 2.0*dt2 + 2.0*dt3 + dt4)/6.0
        dl = (dl1 + 2.0*dl2 + 2.0*dl3 + dl4)/6.0
        dt1 = dt2 = dt3 = dt4 = dl1 = dl2 = dl3 = dl4 = None
        return dt, dl

def trace_err(d1, d2, no):
    # err1 = d_{pp} - N
    # err2 = d_{prqr}/(N-1) - d_{pq}
    err1 = abs(np.trace(d1)-no)
    d2_ = einsum('prqr->pq',d2)
    d2_ /= no - 1
    err2 = np.linalg.norm(d2_-d1)
    d2_ = None
    return err1, err2

def rotate1(A, C):
    # U^{-1}p+U = u+C_{up}, U^{-1}qU = vC_{vq}* 
    # U^{-1}A_{pq}p+qU = A_{pq}C_{up}C_{vq}*u+v
    return np.linalg.multi_dot([C,A,C.T.conj()])

def rotate2(A, C):
    A = einsum('pqrs,up,vq->uvrs',A,C,C)
    return einsum('uvrs,xr,ys->uvxy',A,C.conj(),C.conj())

def kernel_rt_test(mf, t, l, C, w, f0, td, tf, step, RK=4, orb=True):
    eris = ERIs(mf, w, f0, td) # in HF basis
    C = np.array(C, dtype=complex)
    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    eris.rotate(C)
    d1, d2 = compute_rdm12(t, l)
    e = compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+mf.energy_nuc())) 

    no, _, nv, _ = l.shape
    nmo = C.shape[0]
    N = int((tf+step*0.1)/step)

    E = np.zeros(N+1,dtype=complex) 
    tr = np.zeros(2) # trace error
    ec = np.zeros(2) # energy conservation error 
    ehr = 0.0 # ehrenfest error = d<U^{-1}p+qU>/dt - i<U^{-1}[H,p+q]U>
    for i in range(N+1):
        time = i * step 
        eris.rotate(C)
        d1, d2 = compute_rdm12(t, l)
        E[i] = compute_energy(d1, d2, eris, time=None) # <U^{-1}H0U>
        dt, dl = update_RK(t, l, eris, time, step, RK)
        X, F = compute_X(d1, d2, eris, time, no) # F_{qp} = i<[U^{-1}HU,p+q]>
        X = X if orb else np.zeros_like(X, dtype=complex)
        # accumulate analytical error
        tr += np.array(trace_err(d1, d2, no))
        ec += np.array(energy_err(t, l, X, d1, d2, eris, time))
        ehr += np.linalg.norm(compute_der1(t, l, C, X, d1, eris, time)-rotate1(F,C.T.conj()))
        print('time: {:.4f}, EE(mH): {}, X: {}'.format(
              time, (E[i] - E[0]).real*1e3, np.linalg.norm(X)))
        # update 
        t += step * dt
        l += step * dl
        C = np.dot(scipy.linalg.expm(-step*X), C)
    print('trace error: ',tr)
    print('Ehrenfest error: ', ehr)
    print('energy conservation error: ', ec)
    print('imaginary part of energy: ', np.linalg.norm(E.imag))
    return (E - E[0]).real

# not used for now
#def kernel_rt(mf, t, l, U, w, f0, td, tf, step, RK=4, orb=True):
#    U = np.array(U, dtype=complex)
#    t = np.array(t, dtype=complex)
#    l = np.array(l, dtype=complex)
#    no, _, nv, _ = l.shape
#    nmo = U.shape[0]
#    mo0 = mf.mo_coeff.copy()
#    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
#    eris = ERIs(mf, w, f0, td)
#
#    N = int((tf+step*0.1)/step)
#    mus = np.zeros((N+1,3),dtype=complex)  
#    Hmu = np.zeros((N+1,3),dtype=complex)  
#    Es = np.zeros(N+1,dtype=complex)
#
#    d1, d2 = compute_rdms(t, l)
#    mus[0,:] = electric_dipole(d1, mo_coeff, eris)
#    eris.ao2mo(mo_coeff)
#    eris.full_h(time=None)
#    Es[0]  = einsum('pq,qp',eris.h,d1) 
#    Es[0] += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
#    print('check ground state energy: {}'.format(Es[0].real+mf.energy_nuc()))
#    tr = compute_trace(d1, d2, no) 
#    for i in range(N+1):
#        time = i * step 
#        eris.ao2mo(mo_coeff)
#        dt, dl = update_RK(t, l, eris, time, step, RK)
#        d1, d2 = compute_rdms(t, l)
#        X, C = compute_X(d1, d2, eris, time, no) # C_qp = i<[H,p+q]>
#        X = X.copy() if orb else np.zeros_like(X, dtype=complex)
#        # computing observables
#        tr += compute_trace(d1, d2, no) 
#        mus[i,:] = electric_dipole(d1, mo_coeff, eris)
#        Hmu[i,:] = electric_dipole(C, mo_coeff, eris) 
#        eris.full_h(time=None)
#        Es[i]  = einsum('pq,qp',eris.h,d1) 
#        Es[i] += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
#        err = (mus[i,:]-mus[i-1,:])/step - Hmu[i] 
#        print('time: {:.4f}, E(mH): {}, mu: {}, err: {}'.format(
#               time,(Es[i] - Es[0]).real*1e3,(mus[i,:].real-eris.nucl_dip)*1e3, 
#              np.linalg.norm(err)))
#        t += step * dt
#        l += step * dl
#        U = np.dot(U, scipy.linalg.expm(step*X))
#        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
#    print('check trace: {}'.format(tr))
#    print('check E imag: {}'.format(np.linalg.norm(Es.imag)))
#    print('check mu imag: {}'.format(np.linalg.norm(mus.imag)))
#    return mus.real-eris.nucl_dip, (Es - Es[0]).real

class ERIs:
    def __init__(self, mf, w=0.0, f0=np.zeros(3), td=0.0):
        hao = mf.get_hcore()
        eri_ao = mf.mol.intor('int2e_sph')
        mu_ao = mf.mol.intor('int1e_r')
        h1ao = einsum('xuv,x->uv',mu_ao,f0)
        charges = mf.mol.atom_charges()
        coords  = mf.mol.atom_coords()
        self.nucl_dip = einsum('i,ix->x', charges, coords)

        self.w = w
        self.f0 = f0
        self.td = td

        mo_coeff = mf.mo_coeff.copy()
        h0 = einsum('uv,up,vq->pq',hao,mo_coeff,mo_coeff)
        h1 = einsum('uv,up,vq->pq',h1ao,mo_coeff,mo_coeff)
        mu = einsum('xuv,up,vq->xpq',mu_ao,mo_coeff,mo_coeff)
        eri = einsum('uvxy,up,vr->prxy',eri_ao,mo_coeff,mo_coeff)
        eri_ab = einsum('prxy,xq,ys->prqs',eri,mo_coeff,mo_coeff)
        eri_ab = eri_ab.transpose(0,2,1,3)
        eri_aa = eri_ab - eri_ab.transpose(0,1,3,2)

        # underscored tensors always in stationary HF basis
        self.h0_ = sort1((h0,h0))
        self.h1_ = sort1((h1,h1))
        mux = sort1((mu[0,:,:],mu[0,:,:]))
        muy = sort1((mu[1,:,:],mu[1,:,:]))
        muz = sort1((mu[2,:,:],mu[2,:,:]))
        self.mu_ = np.array((mux,muy,muz))
        self.eri_ = sort2((eri_aa, eri_ab, eri_aa),anti=True)

        # U^{-1}HU assuming U = 1
        self.h0 = self.h0_.copy()
        self.h1 = self.h1_.copy()
        self.mu = self.mu_.copy()
        self.eri = self.eri_.copy()

        hao = h1ao = h0 = h1 = None
        mu_ao = mu = mux = muy = muz = None
        eri = eri_ab = eri_aa = None

    def rotate(self, C):
        # compute tnsors of U^{-1}HU
        self.h0 = rotate1(self.h0_, C)
        self.h1 = rotate1(self.h1_, C)
        self.eri = rotate2(self.eri_, C)

    def full_h(self, time=None):
        self.h = full_h(self.h0, self.h1, self.w, self.td, time) 
               
def full_h(h0, h1, w=None, td=None, time=None): 
    # computes H = H0 + H1(t)
    h = h0.copy()
    if time is not None:
        if time < td:
            evlp = math.sin(math.pi*time/td)**2
            osc = math.cos(w*(time-td*0.5))
            h += h1 * evlp * osc
    return h

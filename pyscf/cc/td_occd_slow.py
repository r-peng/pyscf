import numpy as np
import scipy, math
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

def compute_res_t(t, eris):
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

def compute_res_l(t, l, eris):
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

def update_amps(t, l, eris):
    T, L = compute_res(t, l, eris)
    return -1j*T, 1j*L

def compute_res(t, l, eris):
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
    return dt, dl

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

def compute_X(d1, d2, eris, no):
    nmo = d1.shape[0]
    nv = nmo - no
    A  = einsum('vp,qu->uvpq',np.eye(nmo),d1)
#    A -= einsum('qu,vp->uvpq',np.eye(nmo),d1)
    A -= A.transpose(1,0,3,2).conj()
    Aovvo = A[:no,no:,no:,:no].copy()

    C  = einsum('vp,pu->uv',d1,eris.h)
#    C -= einsum('pu,vp->uv',d1,eris.h)
    C += 0.5 * einsum('pqus,vspq->uv',eris.eri,d2)
#    C -= 0.5 * einsum('vqrs,rsuq->uv',eris.eri,d2)
    C -= C.T.conj()
    Cov = C[:no,no:].copy()

    Aovvo = Aovvo.reshape(no*nv,nv*no)
    Cov = Cov.reshape(no*nv)
    
    iXvo = np.dot(np.linalg.inv(Aovvo),Cov)
    iXvo = iXvo.reshape(nv,no)
    
    iX = np.zeros((nmo,nmo),dtype=complex)
    iX[:no,no:] = iXvo.T.conj()
    iX[no:,:no] = iXvo.copy()
    return 1j*iX, 1j*C.T 

def compute_energy(d1, d2, eris):
    e  = einsum('pq,qp',eris.h,d1)
    e += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
    return e.real

def ao2mo(Aao, mo_coeff):
    moa, mob = mo_coeff
    Aa = einsum('uv,up,vq->pq',Aao,moa.conj(),moa)
    Ab = einsum('uv,up,vq->pq',Aao,mob.conj(),mob)
    return sort1((Aa,Ab))

def update_RK4(t, l, eris, time, step, RK4):
    eris.full_h(time)
    dt1, dl1 = update_amps(t, l, eris)
    if not RK4:
        return dt1, dl1
    else: 
        eris.full_h(time+step*0.5)
        dt2, dl2 = update_amps(t + dt1*step*0.5, l + dl1*step*0.5, eris) 
        dt3, dl3 = update_amps(t + dt2*step*0.5, l + dl2*step*0.5, eris) 
        eris.full_h(time+step)
        dt4, dl4 = update_amps(t + dt3*step, l + dl3*step, eris) 
        dt = (dt1 + 2.0*dt2 + 2.0*dt3 + dt4)/6.0
        dl = (dl1 + 2.0*dl2 + 2.0*dl3 + dl4)/6.0
        return dt, dl

def kernel_rt_test(mf, t, l, U, w, f0, td, ts, RK4=True, orb=True):
    U = np.array(U, dtype=complex)
    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    no, _, nv, _ = l.shape
    nmo = U.shape[0]
    mo0 = mf.mo_coeff.copy()
    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    eris = ERIs(mf, w, f0, td)

    N = len(ts)
    d1s = np.zeros((N,nmo,nmo),dtype=complex)  
    d0s = np.zeros((N,nmo,nmo),dtype=complex)  

    d1, d2 = compute_rdms(t, l)
    d1s[0,:,:] = d1.copy()
    d0s[0,:,:] = einsum('qp,vq,up->vu',d1,U,U.conj()) # in stationary HF basis
    tr = abs(np.trace(d1)-no)
    for i in range(1,N):
        time = ts[i]
        step = ts[i] - ts[i-1]
        eris.ao2mo(mo_coeff)
        dt, dl = update_RK4(t, l, eris, time, step, RK4)
        d1, d2 = compute_rdms(t, l)
        eris.full_h(time)
        X, C = compute_X(d1, d2, eris, no) # C_qp = i<[H,p+q]>
        # computing observables
        d1s[i,:,:] = d1.copy()
        d0s[i,:,:] = einsum('qp,vq,up->vu',d1,U,U.conj()) # in stationary HF basis
        LHS = (d1s[i,:,:]-d1s[i-1,:,:])/step
        LHS0 = (d0s[i,:,:]-d0s[i-1,:,:])/step
        C0 = einsum('qp,vq,up->vu',C,U,U.conj()) # in stationary HF basis
        tmp  = einsum('rp,qr->qp',X,d1)
        tmp -= einsum('qr,rp->qp',X,d1)
        RHS = C + tmp
        error = LHS-RHS
#
        LHS_  = einsum('vu,up,vq->qp',LHS0,U,U.conj())
        dU = np.dot(U, X)
        tmp_  = einsum('vu,up,vq->qp',d0s[i,:,:],dU,U.conj())
        tmp_ += einsum('vu,up,vq->qp',d0s[i,:,:],U,dU.conj())
        LHS_ += tmp_.copy()
        diff = LHS - LHS_
#
        print('time: {:.4f}, err oo: {}, vv: {}, ov: {}, vo: {}, diff oo: {}, vv: {}, ov: {}, vo: {}, err0: {}, X: {}'.format(time,
              np.linalg.norm(error[:no,:no]), np.linalg.norm(error[no:,no:]), 
              np.linalg.norm(RHS[:no,no:]), np.linalg.norm(RHS[no:,:no]),
              np.linalg.norm(diff[:no,:no]), np.linalg.norm(diff[no:,no:]), 
              np.linalg.norm(diff[:no,no:]), np.linalg.norm(diff[no:,:no]),
              np.linalg.norm(LHS0-C0), np.linalg.norm(X)))
        if np.linalg.norm(error) > 1.0:
            print('diverging error!')
            break
        t += step * dt
        l += step * dl
        X = X.copy() if orb else np.zeros_like(X, dtype=complex)
        U = np.dot(U, scipy.linalg.expm(step*X))
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    print('check trace: {}'.format(tr))

def kernel_rt(mf, t, l, U, w, f0, td, ts, RK4=True):
    U = np.array(U, dtype=complex)
    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    no, _, nv, _ = l.shape
    mo0 = mf.mo_coeff.copy()
    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    eris = ERIs(mf, w, f0, td)

    N = len(ts)
    mus = np.zeros((N,3),dtype=complex)  
    Hmu = np.zeros((N,3),dtype=complex)  
    Es = np.zeros(N,dtype=complex)

    d1, d2 = compute_rdms(t, l)
    mux = ao2mo(eris.mu_ao[0,:,:], mo_coeff)
    muy = ao2mo(eris.mu_ao[1,:,:], mo_coeff)
    muz = ao2mo(eris.mu_ao[2,:,:], mo_coeff)
    mus[0,0] = einsum('pq,qp',mux,d1)
    mus[0,1] = einsum('pq,qp',muy,d1)
    mus[0,2] = einsum('pq,qp',muz,d1)
    eris.ao2mo(mo_coeff)
    eris.full_h(time=0.0)
    Es[0]  = einsum('pq,qp',eris.h,d1) 
    Es[0] += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
    for i in range(1,N):
        time = ts[i]
        step = ts[i] - ts[i-1]
        eris.ao2mo(mo_coeff)
        dt, dl = update_RK4(t, l, eris, time, step, RK4)
        d1, d2 = compute_rdms(t, l)
        eris.full_h(time)
        X, C = compute_X(d1, d2, eris, no) # C_qp = i<[H,p+q]>
        # computing observables
        mux = ao2mo(eris.mu_ao[0,:,:], mo_coeff)
        muy = ao2mo(eris.mu_ao[1,:,:], mo_coeff)
        muz = ao2mo(eris.mu_ao[2,:,:], mo_coeff)
        mus[i,0] = einsum('pq,qp',mux,d1)
        mus[i,1] = einsum('pq,qp',muy,d1)
        mus[i,2] = einsum('pq,qp',muz,d1)
        Hmu[i,0] = einsum('pq,qp',mux,C)
        Hmu[i,1] = einsum('pq,qp',muy,C)
        Hmu[i,2] = einsum('pq,qp',muz,C)
        Es[i]  = einsum('pq,qp',eris.h,d1) 
        Es[i] += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
        error = (mus[i,:]-mus[i-1,:])/step - Hmu[i] 
        print('time: {:.4f}, ehrenfest: {}, imag: {}, E.imag: {}'.format(time,
              np.linalg.norm(error), np.linalg.norm(mus[i,:].imag), Es[i].imag))
        t += step * dt
        l += step * dl
        U = np.dot(U, scipy.linalg.expm(step*X))
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    return mus, Es

class ERIs:
    def __init__(self, mf, w=0.0, f0=np.zeros(3), td=0.0):
        self.hao = mf.get_hcore()
        self.eri_ao = mf.mol.intor('int2e_sph')
        self.mu_ao = mf.mol.intor('int1e_r')
        self.h1ao = einsum('xuv,x->uv',self.mu_ao,f0)
#        self.mo0 = mf.mo_coeff

        self.w = w
        self.f0 = f0
        self.td = td

    def ao2mo(self, mo_coeff):
        moa, mob = mo_coeff 
    
        self.h0 = ao2mo(self.hao, mo_coeff)
        self.h1 = ao2mo(self.h1ao, mo_coeff)
    
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

    def full_h(self, time=0.0):
        self.h = self.h0.copy()
        if time < self.td:
            evlp = math.sin(math.pi*time/self.td)**2
            osc = math.sin(self.w*time)
            self.h += self.h1 * evlp * osc
            

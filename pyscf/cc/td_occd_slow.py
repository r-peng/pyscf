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

def update_amps(t, l, eris, time=None):
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

    doo = dvv = doooo = doovv = dvvoo = dovvo = dovov = dvvvv = None
    if symm:
        d1 = 0.5 * (d1 + d1.T.conj())
        d2 = 0.5 * (d2 + d2.transpose(2,3,0,1).conj())
    return d1, d2

def compute_X(d1, d2, eris, time, no):
    eris.full_h(time)

    nmo = d1.shape[0]
    nv = nmo - no
    A  = einsum('vp,qu->uvpq',np.eye(nmo),d1)
    A -= A.transpose(1,0,3,2).conj()
    Aovvo = A[:no,no:,no:,:no].copy()
    A = None

    C  = einsum('vp,pu->uv',d1,eris.h)
    C += 0.5 * einsum('pqus,vspq->uv',eris.eri,d2)
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
    Aa = einsum('uv,up,vq->pq',Aao,mo_coeff[0].conj(),mo_coeff[0])
    Ab = einsum('uv,up,vq->pq',Aao,mo_coeff[1].conj(),mo_coeff[1])
    return sort1((Aa,Ab))

def update_RK_(t, l, eris, time, h, RK):
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

def update_RK(t, l, eris, time, h, RK):
    no = l.shape[0]
    dt1, dl1 = update_amps(t, l, eris, time)
    d1_, d2_ = compute_rdms(t, l)
    X1, C = compute_X(d1_, d2_, eris, time, no) # C_qp = i<[H,p+q]>
    if RK == 1:
        return dt1, dl1, X1, X1, C, d1_, d2_
    if RK == 2:
        dt2, dl2 = update_amps(t + dt1*h*0.5, l + dl1*h*0.5, eris, time+h*0.5) 
        d1, d2 =  compute_rdms(t + dt1*h*0.5, l + dl1*h*0.5)
        X2, _ = compute_X(d1, d2, eris, time+h*0.5, no) # C_qp = i<[H,p+q]>
        return dt2, dl2, X2, X1, C, d1_, d2_
    if RK == 4: 
        dt2, dl2 = update_amps(t + dt1*h*0.5, l + dl1*h*0.5, eris, time+h*0.5) 
        d1, d2 =  compute_rdms(t + dt1*h*0.5, l + dl1*h*0.5)
        X2, _ = compute_X(d1, d2, eris, time+h*0.5, no) # C_qp = i<[H,p+q]>
        dt3, dl3 = update_amps(t + dt2*h*0.5, l + dl2*h*0.5, eris, time+h*0.5) 
        d1, d2 =  compute_rdms(t + dt2*h*0.5, l + dl2*h*0.5)
        X3, _ = compute_X(d1, d2, eris, time+h*0.5, no) # C_qp = i<[H,p+q]>
        dt4, dl4 = update_amps(t + dt3*h, l + dl3*h, eris, time+h) 
        d1, d2 =  compute_rdms(t + dt3*h, l + dl3*h)
        X4, _ = compute_X(d1, d2, eris, time+h, no) # C_qp = i<[H,p+q]>
        dt = (dt1 + 2.0*dt2 + 2.0*dt3 + dt4)/6.0
        dl = (dl1 + 2.0*dl2 + 2.0*dl3 + dl4)/6.0
        X  = ( X1 + 2.0* X2 + 2.0* X3 +  X4)/6.0
        dt1 = dt2 = dt3 = dt4 = None
        dl1 = dl2 = dl3 = dl4 = None
        X2 = X3 = X4 = d1 = d2 = None
        return dt, dl, X, X1, C, d1_, d2_

def electric_dipole(d1, mo_coeff, eris):
    mux = ao2mo(eris.mu_ao[0,:,:], mo_coeff)
    muy = ao2mo(eris.mu_ao[1,:,:], mo_coeff)
    muz = ao2mo(eris.mu_ao[2,:,:], mo_coeff)
    mux = einsum('pq,qp',mux,d1)
    muy = einsum('pq,qp',muy,d1)
    muz = einsum('pq,qp',muz,d1)
    return np.array((mux, muy, muz))

def compute_trace(d1, d2, no):
    tr1 = abs(np.trace(d1)-no)
    d2_ = einsum('prqr->pq',d2)
    d2_ /= no - 1
    tr2 = np.linalg.norm(d2_-d1)
    d2_ = None
    return np.array((tr1, tr2))

def compute_derivative(f, step):
    N = len(f)
    def der(i):
        if i >= 2 and i <= N-3: # 4 point
            return (-f[i+2]+8*f[i+1]-8*f[i-1]+f[i-2])/(12*step)
        if i == 1 or i == N-2:
            return (f[i+1] - f[i-1])/(2*step)
        if i == N-1:
            return (f[i] - f[i-1])/step
        if i == 0: 
            return np.zeros_like(f[0])
    df = np.zeros_like(f,dtype=complex) 
    for i in range(len(f)):
        df[i] = der(i)
    return df

def compute_energy(d1, d2, eris, time=None):
    eris.full_h(time)
    E  = einsum('pq,qp',eris.h,d1) 
    E += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
    return E

def kernel_rt_test(mf, t, l, U, w, f0, td, tf, step, RK=4, orb=True):
    U = np.array(U, dtype=complex)
    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    no, _, nv, _ = l.shape
    nmo = U.shape[0]
    mo0 = mf.mo_coeff.copy()
    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    eris = ERIs(mf, w, f0, td)

    N = int((tf+step*0.1)/step)
    d1s = [] 
    d0s = []
    RHS = []
    RHS0 = []
    E = [] 

    d1, d2 = compute_rdms(t, l)
    eris.ao2mo(mo_coeff)
    e = compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+mf.energy_nuc())) 
    tr = compute_trace(d1, d2, no) 
    for i in range(N+1):
        time = i * step 
        eris.ao2mo(mo_coeff)
        dt, dl = update_RK_(t, l, eris, time, step, RK)
        d1, d2 = compute_rdms(t, l)
        X, C = compute_X(d1, d2, eris, time, no) # C_qp = i<[H,p+q]>
        X1 = X.copy()
#        dt, dl, X, X1, C, d1, d2 = update_RK(t, l, eris, time, step, RK)
        X = X.copy() if orb else np.zeros_like(X, dtype=complex)
        X1 = X1.copy() if orb else np.zeros_like(X, dtype=complex)
        # computing observables
        tr += compute_trace(d1, d2, no) 
        E.append(compute_energy(d1, d2, eris, time=None))
        d1s.append(d1.copy())
        d0s.append(einsum('qp,vq,up->vu',d1,U,U.conj())) # in stationary HF basis
        LHS = (d1s[i]-d1s[i-1])/step
        LHS0 = (d0s[i]-d0s[i-1])/step
        tmp  = einsum('rp,qr->qp',X1,d1)
        tmp -= einsum('qr,rp->qp',X1,d1)
        RHS.append(C + tmp)
        RHS0.append(einsum('qp,vq,up->vu',C,U,U.conj())) # in stationary HF basis
        err = LHS-RHS[i]
#
#        LHS_  = einsum('vu,up,vq->qp',LHS0,U,U.conj())
#        dU = np.dot(U, X)
#        tmp_  = einsum('vu,up,vq->qp',d0s[i,:,:],dU,U.conj())
#        tmp_ += einsum('vu,up,vq->qp',d0s[i,:,:],U,dU.conj())
#        LHS_ += tmp_.copy()
#        diff = LHS - LHS_
#
#        print('time: {:.4f}, err oo: {}, vv: {}, ov: {}, vo: {}, diff oo: {}, vv: {}, ov: {}, vo: {}, err0: {}, X: {}'.format(time,
#              np.linalg.norm(error[:no,:no]), np.linalg.norm(error[no:,no:]), 
#              np.linalg.norm(RHS[:no,no:]), np.linalg.norm(RHS[no:,:no]),
#              np.linalg.norm(diff[:no,:no]), np.linalg.norm(diff[no:,no:]), 
#              np.linalg.norm(diff[:no,no:]), np.linalg.norm(diff[no:,:no]),
#              np.linalg.norm(LHS0-C0), np.linalg.norm(X)))
#
        print('time: {:.4f}, E(mH): {}, err: {}, X: {}'.format(
              time, (E[-1] - E[0]).real*1e3,
              np.linalg.norm(err), np.linalg.norm(X)))
        if np.linalg.norm(err) > 1.0:
            print('diverging error!')
            break
        t += step * dt
        l += step * dl
        U = np.dot(U, scipy.linalg.expm(step*X))
#        U += step * np.dot(U,X)
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    E = np.array(E)
    print('check trace: {}'.format(tr))
    print('check E imag: {}'.format(np.linalg.norm(E.imag)))
    print('check error')
    LHS = compute_derivative(d1s, step)
    LHS0 = compute_derivative(d0s, step)
    for i in range(len(LHS)):
        print('err: {}, err0: {}'.format(
              np.linalg.norm(LHS[i]-RHS[i]),np.linalg.norm(LHS0[i]-RHS0[i])))
    return (E - E[0]).real

def kernel_rt(mf, t, l, U, w, f0, td, tf, step, RK=4, orb=True):
    U = np.array(U, dtype=complex)
    t = np.array(t, dtype=complex)
    l = np.array(l, dtype=complex)
    no, _, nv, _ = l.shape
    nmo = U.shape[0]
    mo0 = mf.mo_coeff.copy()
    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    eris = ERIs(mf, w, f0, td)

    N = int((tf+step*0.1)/step)
    mus = np.zeros((N+1,3),dtype=complex)  
    Hmu = np.zeros((N+1,3),dtype=complex)  
    Es = np.zeros(N+1,dtype=complex)

    d1, d2 = compute_rdms(t, l)
    mus[0,:] = electric_dipole(d1, mo_coeff, eris)
    eris.ao2mo(mo_coeff)
    eris.full_h(time=None)
    Es[0]  = einsum('pq,qp',eris.h,d1) 
    Es[0] += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
    print('check ground state energy: {}'.format(Es[0].real+mf.energy_nuc()))
    tr = compute_trace(d1, d2, no) 
    for i in range(N+1):
        time = i * step 
        eris.ao2mo(mo_coeff)
        dt, dl = update_RK(t, l, eris, time, step, RK)
        d1, d2 = compute_rdms(t, l)
        X, C = compute_X(d1, d2, eris, time, no) # C_qp = i<[H,p+q]>
        X = X.copy() if orb else np.zeros_like(X, dtype=complex)
        # computing observables
        tr += compute_trace(d1, d2, no) 
        mus[i,:] = electric_dipole(d1, mo_coeff, eris)
        Hmu[i,:] = electric_dipole(C, mo_coeff, eris) 
        eris.full_h(time=None)
        Es[i]  = einsum('pq,qp',eris.h,d1) 
        Es[i] += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
        err = (mus[i,:]-mus[i-1,:])/step - Hmu[i] 
        print('time: {:.4f}, E(mH): {}, mu: {}, err: {}'.format(
               time,(Es[i] - Es[0]).real*1e3,(mus[i,:].real-eris.nucl_dip)*1e3, 
              np.linalg.norm(err)))
        t += step * dt
        l += step * dl
        U = np.dot(U, scipy.linalg.expm(step*X))
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    print('check trace: {}'.format(tr))
    print('check E imag: {}'.format(np.linalg.norm(Es.imag)))
    print('check mu imag: {}'.format(np.linalg.norm(mus.imag)))
    return mus.real-eris.nucl_dip, (Es - Es[0]).real

class ERIs:
    def __init__(self, mf, w=0.0, f0=np.zeros(3), td=0.0):
        self.hao = mf.get_hcore()
        self.eri_ao = mf.mol.intor('int2e_sph')
        self.mu_ao = mf.mol.intor('int1e_r')
        self.h1ao = einsum('xuv,x->uv',self.mu_ao,f0)
        charges = mf.mol.atom_charges()
        coords  = mf.mol.atom_coords()
        self.nucl_dip = einsum('i,ix->x', charges, coords)

        self.w = w
        self.f0 = f0
        self.td = td

#
        sao = mf.get_ovlp()
        nao = sao.shape[0]
        mo0 = mf.mo_coeff
        print(np.linalg.norm(np.eye(nao)-einsum('up,uv,vq->pq',mo0,sao,mo0)))
        exit()
#

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
        eri_aa = eri_ab = eri_bb = eri = moa = mob = None

    def full_h(self, time=None):
        self.h = self.h0.copy()
        if time is not None:
            if time < self.td:
                evlp = math.sin(math.pi*time/self.td)**2
                osc = math.cos(self.w*(time-self.td*0.5))
                self.h += self.h1 * evlp * osc
                

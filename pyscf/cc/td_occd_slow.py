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

def update_t(t, eris):
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

def update_l(t, l, eris):
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
    return -1j*dt, 1j*dl

def compute_gamma1(t, l): # normal ordered, asymmetric
#    dvv = 0.5 * einsum('ikac,bcik->ab',l,t)
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

def compute_X_(d1, d2, eris, res_t, res_l, t, l):
    nmo = d1.shape[0]
    no = res_l.shape[0]
    nv = nmo - no
    A  = einsum('vp,qu->uvpq',np.eye(nmo),d1)
    A -= einsum('qu,vp->uvpq',np.eye(nmo),d1)
#    print('A symm: {}'.format(np.linalg.norm(A+A.transpose(1,0,3,2).conj())))
#    exit()

    C  = einsum('vp,pu->uv',d1,eris.h)
    C -= einsum('pu,vp->uv',d1,eris.h)
    C += 0.5 * einsum('pqus,vspq->uv',eris.eri,d2)
    C -= 0.5 * einsum('vqrs,rsuq->uv',eris.eri,d2)
#    print('C symm: {}'.format(np.linalg.norm(C+C.T.conj())))

    B = np.zeros((nmo,nmo),dtype=complex)
    B[:no,:no] += einsum('abuj,vjab->uv',res_t,l) # res_t = i*dt
    B[:no,:no] -= einsum('abvj,ujab->uv',res_t,l).conj()
    B[no:,no:] -= einsum('vbij,ijub->uv',res_t,l)
    B[no:,no:] += einsum('ubij,ijvb->uv',res_t,l).conj()

    B[:no,:no] -= einsum('vjab,abuj->uv',res_l,t) # res_l = -i*dl
    B[:no,:no] += einsum('ujab,abvj->uv',res_l,t).conj() 
    B[no:,no:] += einsum('ijub,vbij->uv',res_l,t)
    B[no:,no:] -= einsum('ijvb,ubij->uv',res_l,t).conj()
#    print('B symm: {}'.format(np.linalg.norm(B+B.T.conj())))

    RHS = C - B
#    print('RHS symm: {}'.format(np.linalg.norm(RHS+RHS.T.conj())))

    Aovvo = A[:no,no:,no:,:no].copy().reshape(no*nv,nv*no)
    Avoov = A[no:,:no,:no,no:].copy().reshape(nv*no,no*nv)
    Aoooo = A[:no,:no,:no,:no].copy().reshape(no*no,no*no)
    Avvvv = A[no:,no:,no:,no:].copy().reshape(nv*nv,nv*nv)
#    print('Aovvo det: {}'.format(abs(np.linalg.det(Aovvo))))
#    print('Avoov det: {}'.format(abs(np.linalg.det(Avoov))))
#    print('Aoooo det: {}'.format(abs(np.linalg.det(Aoooo))))
#    print('Avvvv det: {}'.format(abs(np.linalg.det(Avvvv))))
    RHSov = RHS[:no,no:].copy().reshape(no*nv)
    RHSvo = RHS[no:,:no].copy().reshape(nv*no)
    RHSoo = RHS[:no,:no].copy().reshape(no*no)
    RHSvv = RHS[no:,no:].copy().reshape(nv*nv)
    
    Xvo = 1j * np.dot(np.linalg.inv(Aovvo),RHSov)
    Xov = 1j * np.dot(np.linalg.inv(Avoov),RHSvo)
   
    X = np.zeros((nmo,nmo),dtype=complex)
    X[:no,no:] = Xov.reshape(no,nv).copy()
    X[no:,:no] = Xvo.reshape(nv,no).copy()
#    print('X symm: {}'.format(np.linalg.norm(X+X.T.conj())))
    check = einsum('uvpq,pq->uv',A,-1j*X) + B - C
    print('ov/vo: {}'.format(np.linalg.norm(check[:no,no:])+np.linalg.norm(check[no:,:no])))
    print('oo: {}'.format(np.linalg.norm(check[:no,:no])))
    print('vv: {}'.format(np.linalg.norm(check[no:,no:])))

#    Xvv = scipy.linalg.solve(Avvvv,RHSvv) 
#    Xvv, _, _, _ = scipy.linalg.lstsq(Avvvv,RHSvv) 
#    Xoo, _, _, _ = scipy.linalg.lstsq(Aoooo,RHSoo)
#    Xvv = 1j * Xvv.reshape(nv,nv)
#    Xoo = 1j * Xoo.reshape(no,no)
#    print('Xvv symm: {}'.format(np.linalg.norm(Xvv+Xvv.T.conj())))
#    print('Xoo symm: {}'.format(np.linalg.norm(Xoo+Xoo.T.conj())))
    return X, 1j*B.T, 1j*C.T 

def compute_X(d1, d2, eris, no):
    nmo = d1.shape[0]
    nv = nmo - no
    A  = einsum('vp,qu->uvpq',np.eye(nmo),d1)
    A -= einsum('qu,vp->uvpq',np.eye(nmo),d1)
    Aovvo = A[:no,no:,no:,:no].copy()

    C  = einsum('vp,pu->uv',d1,eris.h)
    C -= einsum('pu,vp->uv',d1,eris.h)
    C += 0.5 * einsum('pqus,vspq->uv',eris.eri,d2)
    C -= 0.5 * einsum('vqrs,rsuq->uv',eris.eri,d2)
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

def update_RK4(t, l, eris, step, RK4):
    dt1, dl1 = update_amps(t, l, eris)
    if not RK4:
        return dt1, dl1 
    else: 
        dt2, dl2 = update_amps(t + dt1*step*0.5, l + dl1*step*0.5, eris) 
        dt3, dl3 = update_amps(t + dt2*step*0.5, l + dl2*step*0.5, eris) 
        dt4, dl4 = update_amps(t + dt3*step, l + dl3*step, eris) 
        dt = (dt1 + 2.0*dt2 + 2.0*dt3 + dt4)/6.0
        dl = (dl1 + 2.0*dl2 + 2.0*dl3 + dl4)/6.0
        return dt, dl

def kernel_rt_test(mf, t, l, U, w, f0, tp, tf, step, RK4=True, orb=True):
    nao = mf.mol.nao_nr()
    mu_ao = mf.mol.intor('int1e_r')
    hao = einsum('xuv,x->uv',mu_ao,f0)

    td = 2 * int(tp/step)
    maxiter = int(tf/step)
    no, _, nv, _ = l.shape
    mo0 = mf.mo_coeff.copy()
    U = np.array(U, dtype=complex)
    X = np.zeros((no+nv,)*2,dtype=complex)
    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    eris = ERIs(mf)

    d1_old, d2 = compute_rdms(t, l)
    d0_old = einsum('qp,vq,up->vu',d1_old,U,U.conj()) # in stationary HF basis
    tr = abs(np.trace(d1_old)-no)
    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
            osc = math.sin(w*i*step) 
            eris.h += ao2mo(hao, mo_coeff) * osc * evlp
        dt, dl = update_RK4(t, l, eris, step, RK4=RK4)
        d1, d2 = compute_rdms(t, l)
#        X, C = compute_X(d1, d2, eris, no) # C_qp = i<[H,p+q]>
        X, B, C = compute_X_(d1, d2, eris, 1j*dt,-1j*dl, t, l) # C_qp = i<[H,p+q]>
        t += step * dt
        l += step * dl
        d0 = einsum('qp,vq,up->vu',d1,U,U.conj()) # in stationary HF basis
        dd1, d1_old = d1-d1_old, d1.copy()
        dd0, d0_old = d0-d0_old, d0.copy()
        LHS = dd1/step
        LHS0 = dd0/step
        C0 = einsum('qp,vq,up->vu',C,U,U.conj()) # in stationary HF basis
        tmp  = einsum('rp,qr->qp',X,d1)
        tmp -= einsum('qr,rp->qp',X,d1)
        RHS = C + tmp
#
        LHS_  = einsum('vu,up,vq->qp',LHS0,U,U.conj())
        dU = np.dot(U, X)
        tmp_  = einsum('vu,up,vq->qp',d0,dU,U.conj())
        tmp_ += einsum('vu,up,vq->qp',d0,U,dU.conj())
        LHS_ += tmp_.copy()
        diff = LHS - LHS_
        print('diff oo, vv, ov, vo: ', np.linalg.norm(diff[:no,:no]), np.linalg.norm(diff[no:,no:]), np.linalg.norm(diff[:no,no:]), np.linalg.norm(diff[no:,:no]))
#
        error = LHS-RHS
        if orb:
            U = np.dot(U, scipy.linalg.expm(step*X))
#            U += dU * step 
            mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
            print('time: {:.4f}, d1(oo): {}, d1(vv): {}, d1(ov/vo): {}, d0: {}, normX: {}'.format(i*step,np.linalg.norm(error[:no,:no]), np.linalg.norm(error[no:,no:]), 
                   np.linalg.norm(RHS[:no,no:])+np.linalg.norm(RHS[no:,:no]), 
                   np.linalg.norm(LHS0-C0), np.linalg.norm(X)))
        else:
            print('time: {:.4f}, d1: {}, d1(ov/vo): {}, d0: {}'.format(
                   i*step, np.linalg.norm(error), 
                   np.linalg.norm(RHS[:no,no:])+np.linalg.norm(RHS[no:,:no]), 
                   np.linalg.norm(LHS0-C0))) 
        if np.linalg.norm(error) > 1.0:
            print('diverging error!')
            break
        tr += abs(np.trace(d1_old)-no)
    print('check trace: {}'.format(tr))

def kernel_rt(mf, t, l, U, w, f0, tp, tf, step, RK4=True):
    nao = mf.mol.nao_nr()
    mu_ao = mf.mol.intor('int1e_r')
    hao = einsum('xuv,x->uv',mu_ao,f0)

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
            osc = math.sin(w*i*step) 
            eris.h += ao2mo(hao, mo_coeff) * evlp * osc
        mux_mo = ao2mo(mu_ao[0,:,:], mo_coeff)
        muy_mo = ao2mo(mu_ao[1,:,:], mo_coeff)
        muz_mo = ao2mo(mu_ao[2,:,:], mo_coeff)
        dt, dl = update_RK4(t, l, eris, step, RK4=RK4)
        d1, d2 = compute_rdms(t, l)
        X, C = compute_X(d1, d2, eris, no) # C_qp = i<[H,p+q]>
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
        print('time: {:.4f}, ehrenfest: {}, imag: {}, E.imag: {}'.format(i*step, abs(error), imag, E[i].imag))
#        print('mux: {}, muy: {}, muz: {}'.format(mux[i+1].real,muy[i+1].real,muz[i+1].real))
    return mux, muy, muz, E


class ERIs:
    def __init__(self, mf):
        self.hao = mf.get_hcore()
        self.eri_ao = mf.mol.intor('int2e_sph')

    def ao2mo(self, mo_coeff):
        moa, mob = mo_coeff
        nmoa, nmob = moa.shape[0], mob.shape[0]
    
        ha = einsum('uv,up,vq->pq',self.hao,moa.conj(),moa)
        hb = einsum('uv,up,vq->pq',self.hao,mob.conj(),mob)
        self.h = sort1((ha,hb))
    
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

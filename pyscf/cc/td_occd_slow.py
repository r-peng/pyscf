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

def update_t(t, eris, X):
    no = t.shape[3]
    eri = eris.eri.copy()
    f  = eris.h.copy() - 1j*X
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

def update_l(t, l, eris, X):
    no = t.shape[3]
    eri = eris.eri.copy()
    f  = eris.h.copy() - 1j*X
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

def update_amps(t, l, eris, X, update_X=False):
    no = t.shape[3]
    eri = eris.eri.copy()
    f  = eris.h.copy() - 1j*X
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

    d1 = d2 = C = None
    if update_X:
        d1, d2 = compute_rdms(t, l)
        X, C = compute_X(d1, d2, eris, no) # C_qp = i<[H,p+q]>
    return -1j*dt, 1j*dl, X, C, d1, d2

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

#def compute_kappa(d1, d2, eris, res_t, res_l, t, l):
#    nmo = d1.shape[0]
#    no = res_l.shape[0]
#    nv = nmo - no
#    A  = einsum('vp,qu->uvpq',np.eye(nmo),d1)
#    A -= einsum('qu,vp->uvpq',np.eye(nmo),d1)
#    Aovvo = A[:no,no:,no:,:no].copy()
#    Avoov = A[no:,:no,:no,no:].copy()
#    Aoooo = A[:no,:no,:no,:no].copy()
#    Avvvv = A[no:,no:,no:,no:].copy()
##    print('A symm: {}'.format(np.linalg.norm(A+A.transpose(1,0,3,2).conj())))
##    exit()
#
#    C  = einsum('vp,pu->uv',d1,eris.h)
#    C -= einsum('pu,vp->uv',d1,eris.h)
#    C += 0.5 * einsum('pqus,vspq->uv',eris.eri,d2)
#    C -= 0.5 * einsum('vqrs,rsuq->uv',eris.eri,d2)
#    Cov = C[:no,no:].copy()
#    Cvo = C[no:,:no].copy()
##    print('C symm: {}'.format(np.linalg.norm(C+C.T.conj())))
#
#    B = np.zeros((nmo,nmo),dtype=complex)
#    B[:no,:no] += einsum('abuj,vjab->uv',res_t,l)
#    B[:no,:no] -= einsum('abvj,ujab->uv',res_t,l).conj()
#    B[no:,no:] -= einsum('vbij,ijub->uv',res_t,l)
#    B[no:,no:] += einsum('ubij,ijvb->uv',res_t,l).conj()
#
#    B[:no,:no] -= einsum('vjab,abuj->uv',res_l,t)
#    B[:no,:no] += einsum('ujab,abvj->uv',res_l,t).conj() 
#    B[no:,no:] += einsum('ijub,vbij->uv',res_l,t)
#    B[no:,no:] -= einsum('ijvb,ubij->uv',res_l,t).conj()
##    print('B symm: {}'.format(np.linalg.norm(B+B.T.conj())))
#
#    Aovvo = Aovvo.reshape(no*nv,nv*no)
#    Avoov = Avoov.reshape(nv*no,no*nv)
#    Cov = Cov.reshape(no*nv)
#    Cvo = Cvo.reshape(nv*no)
#    
#    Xvo = 1j * np.dot(np.linalg.inv(Aovvo),Cov)
#    Xov = 1j * np.dot(np.linalg.inv(Avoov),Cvo)
#    Xvo = Xvo.reshape(nv,no)
#    Xov = Xov.reshape(no,nv)
#    
#    X = np.zeros((nmo,nmo),dtype=complex)
#    X[:no,no:] = Xov.copy()
#    X[no:,:no] = Xvo.copy()
##    print('X: {}'.format(np.linalg.norm(X+X.T.conj())))
#    B += einsum('uvpq,pq->uv',A,-1j*X)
#    check = B - C
##    print(np.linalg.norm(check[:no,no:]))
##    print(np.linalg.norm(check[no:,:no]))
#    print('oo: {}'.format(np.linalg.norm(check[:no,:no])))
#    print('vv: {}'.format(np.linalg.norm(check[no:,:no])))
#
#    vv  = einsum('abcd,cd->ab',A[no:,no:,no:,no:],-1j*C[no:,no:])
#    vv += B[no:,no:].copy() 
#    oo  = einsum('ijkl,kl->ij',A[:no,:no,:no,:no],-1j*C[:no,:no])
#    oo += B[:no,:no].copy()
#    print('vv: {}'.format(np.linalg.norm(vv+vv.T.conj())))
#    print('oo: {}'.format(np.linalg.norm(oo+oo.T.conj())))
#    print('vv: {}'.format(np.linalg.norm(vv-C[no:,no:])))
#    print('oo: {}'.format(np.linalg.norm(oo-C[:no,:no])))
#    exit()
#
#    def compute_grad(iX, A, B, C):
#        tmp = einsum('uvpq,pq->uv',A,iX) + B - C
#        grad  = einsum('uvpq,uv->pq',A,tmp.conj())
#        grad -= einsum('vupq,uv->pq',A,tmp)
#        return grad, np.linalg.norm(tmp)
#    grad_oo, res_oo = compute_grad(-1j*C[:no,:no], A[:no,:no,:no,:no], B[:no,:no], C[:no,:no])
#    print(np.linalg.norm(grad_oo-grad_oo.T.conj()), res_oo)
#    grad_vv, res_vv = compute_grad(-1j*C[no:,no:], A[no:,no:,no:,no:], B[no:,no:], C[no:,no:])
#    print(np.linalg.norm(grad_vv-grad_vv.T.conj()), res_vv)
#    maxiter = 20
#    thresh = 1e-6
#    step = 0.05
#    def lstsq(A, B, C):
#        iX = -1j*C
#        for i in range(maxiter): 
#            grad, res = compute_grad(iX, A, B, C)
#            dnorm = np.linalg.norm(grad)
#            print('iter: {}, res: {}, dnorm: {}'.format(i, res, dnorm))
#            if dnorm < thresh:
#                break
#            iX -= step * grad
#        return 1j*iX
#    Xoo = lstsq(A[:no,:no,:no,:no],B[:no,:no],C[:no,:no])
#    Xvv = lstsq(A[no:,no:,no:,no:],B[no:,no:],C[no:,no:])
#    exit()
#    return X, 1j*B.T, 1j*C.T 

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

def update_RK4(t, l, X, eris, step, update_X):
    dt1, dl1, X1, C, d1, d2 = update_amps(t, l, eris, X, update_X) 
    if update_X:
        dt2, dl2, X2, _, _, _ = update_amps(t + dt1*step*0.5, l + dl1*step*0.5, eris, X + X1*step*0.5, update_X) 
        dt3, dl3, X3, _, _, _ = update_amps(t + dt2*step*0.5, l + dl2*step*0.5, eris, X + X2*step*0.5, update_X) 
        dt4, dl4, X4, _, _, _ = update_amps(t + dt3*step, l + dl3*step, eris, X + X3*step, update_X) 
        X = (X1 + 2.0*X2 + 2.0*X3 + X4)/6.0
    else:
        dt2, dl2, _, _, _, _ = update_amps(t + dt1*step*0.5, l + dl1*step*0.5, eris, X, update_X) 
        dt3, dl3, _, _, _, _ = update_amps(t + dt2*step*0.5, l + dl2*step*0.5, eris, X, update_X) 
        dt4, dl4, _, _, _, _ = update_amps(t + dt3*step, l + dl3*step, eris, X, update_X) 

    dt = (dt1 + 2.0*dt2 + 2.0*dt3 + dt4)/6.0
    dl = (dl1 + 2.0*dl2 + 2.0*dl3 + dl4)/6.0
    return dt, dl, X, C, d1, d2

def kernel_rt_test_RK4(mf, t, l, U, w, f0, tp, tf, step, RK4_X=False):
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

    Aao = np.random.rand(nao,nao)
#    Aao += Aao.T
    Amo0 = ao2mo(Aao, (mo0,mo0)) # in stationary HF basis
    d1_old, d2 = compute_rdms(t, l)
    d0_old = einsum('qp,vq,up->vu',d1_old,U,U.conj()) # in stationary HF basis
    Amo = ao2mo(Aao, mo_coeff)
    A_old = einsum('pq,qp',Amo,d1_old)
    A0_old = einsum('pq,qp',Amo0,d0_old)
    tr = abs(np.trace(d1_old)-no)
    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
            osc = math.cos(w*(i*step-tp)) 
            eris.h += ao2mo(hao, mo_coeff) * osc * evlp
        Amo = ao2mo(Aao, mo_coeff)
        if RK4_X: 
            dt, dl, X, C, d1, d2 = update_RK4(t, l, X, eris, step, update_X=True)
        else: 
            dt, dl, _, _, _, _ = update_RK4(t, l, X, eris, step, update_X=False)
            d1, d2 = compute_rdms(t, l)
            X, C = compute_X(d1, d2, eris, no) # C_qp = i<[H,p+q]>
        t += step * dt
        l += step * dl
        d0 = einsum('qp,vq,up->vu',d1,U,U.conj()) # in stationary HF basis
        A = einsum('pq,qp',Amo,d1)
        A0 = einsum('pq,qp',Amo0,d0)
        dd1, d1_old = d1-d1_old, d1.copy()
        dd0, d0_old = d0-d0_old, d0.copy()
        dA, A_old = A-A_old, A
        dA0, A0_old = A0-A0_old, A0
        LHS = dd1/step
        LHS0 = dd0/step
        C0 = einsum('qp,vq,up->vu',C,U,U.conj()) # in stationary HF basis
        HA = einsum('pq,qp',Amo,C)
        HA0 = einsum('pq,qp',Amo0,C0)
        tmp  = einsum('rp,qr->qp',X,d1)
        tmp -= einsum('qr,rp->qp',X,d1)
        RHS = C + tmp
        U = np.dot(U, scipy.linalg.expm(step*X))
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
        print('time: {:.4f}, d1: {}, d0: {}, A: {}, A0: {}, A.imag: {}, normX: {}'.format(
               i*step, np.linalg.norm(LHS-RHS), np.linalg.norm(LHS0-C0), 
               abs(dA/step-HA), abs(dA0/step-HA0), A_old.imag, np.linalg.norm(X)))
#        print('time: {:.4f}, d1: {}, d0: {}, A: {}, A0: {}, A.imag: {}'.format(
#               i*step, np.linalg.norm(LHS-RHS), np.linalg.norm(LHS0-C0), 
#               abs(dA/step-HA), abs(dA0/step-HA0), A_old.imag))
        if np.linalg.norm(LHS-RHS) > 1.0:
            print('diverging error!')
            break
        tr += abs(np.trace(d1_old)-no)
    print('check trace: {}'.format(tr))


def kernel_rt_test(mf, t, l, U, w, f0, tp, tf, step):
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

    Aao = np.random.rand(nao,nao)
#    Aao += Aao.T
    Amo0 = ao2mo(Aao, (mo0,mo0)) # in stationary HF basis
    d1_old, d2 = compute_rdms(t, l)
    d0_old = einsum('qp,vq,up->vu',d1_old,U,U.conj()) # in stationary HF basis
    Amo = ao2mo(Aao, mo_coeff)
    A_old = einsum('pq,qp',Amo,d1_old)
    A0_old = einsum('pq,qp',Amo0,d0_old)
    tr = abs(np.trace(d1_old)-no)
    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        if i <= td:
            evlp = math.sin(math.pi*i/td)**2
            osc = math.cos(w*(i*step-tp)) 
            eris.h += ao2mo(hao, mo_coeff) * osc * evlp
        Amo = ao2mo(Aao, mo_coeff)
        dt = update_t(t, eris, X) # idt
        dl = update_l(t, l, eris, X) # -idl
        t -= 1j * step * dt
        l += 1j * step * dl
        d1, d2 = compute_rdms(t, l)
        d0 = einsum('qp,vq,up->vu',d1,U,U.conj()) # in stationary HF basis
        A = einsum('pq,qp',Amo,d1)
        A0 = einsum('pq,qp',Amo0,d0)
        dd1, d1_old = d1-d1_old, d1.copy()
        dd0, d0_old = d0-d0_old, d0.copy()
        dA, A_old = A-A_old, A
        dA0, A0_old = A0-A0_old, A0
        LHS = dd1/step
        LHS0 = dd0/step
        X, C = compute_X(d1, d2, eris, no) # C_qp = i<[H,p+q]>
        C0 = einsum('qp,vq,up->vu',C,U,U.conj()) # in stationary HF basis
        HA = einsum('pq,qp',Amo,C)
        HA0 = einsum('pq,qp',Amo0,C0)
        tmp  = einsum('rp,qr->qp',X,d1)
        tmp -= einsum('qr,rp->qp',X,d1)
        RHS = C + tmp
#        LHS_  = einsum('vu,up,vq->qp',LHS0,U,U.conj())
#        dU = np.dot(U, X)
#        tmp_  = einsum('vu,up,vq->qp',d0,dU,U.conj())
#        tmp_ += einsum('vu,up,vq->qp',d0,U,dU.conj())
#        LHS_ += tmp_.copy()
#        diff = LHS - LHS_
#        print(np.linalg.norm(diff))
#        print(np.linalg.norm(diff-diff.T.conj()))
#        print(abs(einsum('pq,qp',Amo,diff)))
#        print(abs(einsum('pq,pq',Amo,diff)))
#        print(np.linalg.norm(np.dot(Amo,diff)))
#        print(abs(np.trace(np.dot(Amo,diff))))

        U = np.dot(U, scipy.linalg.expm(step*X))
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
        print('time: {:.4f}, d1: {}, d0: {}, A: {}, A0: {}, A.imag: {}, normX: {}'.format(
               i*step, np.linalg.norm(LHS-RHS), np.linalg.norm(LHS0-C0), 
               abs(dA/step-HA), abs(dA0/step-HA0), A_old.imag, np.linalg.norm(X)))
#        print('time: {:.4f}, d1: {}, d0: {}, A: {}, A0: {}, A.imag: {}'.format(
#               i*step, np.linalg.norm(LHS-RHS), np.linalg.norm(LHS0-C0), 
#               abs(dA/step-HA), abs(dA0/step-HA0), A_old.imag))
        if np.linalg.norm(LHS-RHS) > 1.0:
            print('diverging error!')
            break
        tr += abs(np.trace(d1_old)-no)
    print('check trace: {}'.format(tr))

def kernel_rt_RK4(mf, t, l, U, w, f0, tp, tf, step, RK4_X=False):
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
            osc = math.cos(w*(i*step-tp)) 
            eris.h += ao2mo(hao, mo_coeff) * evlp * osc
        mux_mo = ao2mo(mu_ao[0,:,:], mo_coeff)
        muy_mo = ao2mo(mu_ao[1,:,:], mo_coeff)
        muz_mo = ao2mo(mu_ao[2,:,:], mo_coeff)
        if RK4_X: 
            dt, dl, X, C, d1, d2 = update_RK4(t, l, X, eris, step, update_X=True)
        else: 
            dt, dl, _, _, _, _ = update_RK4(t, l, X, eris, step, update_X=False)
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
        print('time: {:.4f}, ehrenfest: {}, imag: {}, E.imag: {},'.format(i*step, abs(error), imag, E[i].imag))
#        print('mux: {}, muy: {}, muz: {}'.format(mux[i+1].real,muy[i+1].real,muz[i+1].real))
    return mux, muy, muz, E

def kernel_rt(mf, t, l, U, w, f0, tp, tf, step):
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
            osc = math.cos(w*(i*step-tp)) 
            eris.h += ao2mo(hao, mo_coeff) * evlp * osc
        mux_mo = ao2mo(mu_ao[0,:,:], mo_coeff)
        muy_mo = ao2mo(mu_ao[1,:,:], mo_coeff)
        muz_mo = ao2mo(mu_ao[2,:,:], mo_coeff)
        dt = update_t(t, eris, X) # idt
        dl = update_l(t, l, eris, X) # -idl
        t -= 1j * step * dt
        l += 1j * step * dl
        d1, d2 = compute_rdms(t, l)
        mux[i+1] = einsum('pq,qp',mux_mo,d1)
        muy[i+1] = einsum('pq,qp',muy_mo,d1)
        muz[i+1] = einsum('pq,qp',muz_mo,d1)
        e  = einsum('pq,qp',eris.h,d1) 
        e += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
        E[i] = e
        X, C = compute_X(d1, d2, eris, no)
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

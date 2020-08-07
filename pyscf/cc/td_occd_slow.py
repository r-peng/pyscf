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

def compute_kappa(d1, d2, eris, res_t, res_l, t, l):
    nmo = d1.shape[0]
    no = res_l.shape[0]
    nv = nmo - no
    A  = einsum('vp,qu->uvpq',np.eye(nmo),d1)
    A -= einsum('qu,vp->uvpq',np.eye(nmo),d1)
    Aovvo = A[:no,no:,no:,:no].copy()
    Avoov = A[no:,:no,:no,no:].copy()
#    print('A symm: {}'.format(np.linalg.norm(Aovvo+Avoov.transpose(1,0,3,2).conj())))

    C  = einsum('vp,pu->uv',d1,eris.h)
    C -= einsum('pu,vp->uv',d1,eris.h)
    C += 0.5 * einsum('pqus,vspq->uv',eris.eri,d2)
    C -= 0.5 * einsum('vqrs,rsuq->uv',eris.eri,d2)
    Cov = C[:no,no:].copy()
    Cvo = C[no:,:no].copy()
#    print('C symm: {}'.format(np.linalg.norm(C+C.T.conj())))

    B = np.zeros((nmo,nmo),dtype=complex)
    B[:no,:no] += einsum('abuj,vjab->uv',res_t,l)
    B[:no,:no] -= einsum('abvj,ujab->uv',res_t,l).conj()
    B[no:,no:] -= einsum('vbij,ijub->uv',res_t,l)
    B[no:,no:] += einsum('ubij,ijvb->uv',res_t,l).conj()

    B[:no,:no] -= einsum('vjab,abuj->uv',res_l,t)
    B[:no,:no] += einsum('ujab,abvj->uv',res_l,t).conj() 
    B[no:,no:] += einsum('ijub,vbij->uv',res_l,t)
    B[no:,no:] -= einsum('ijvb,ubij->uv',res_l,t).conj()
#    print('B symm: {}'.format(np.linalg.norm(B+B.T.conj())))

    Aovvo = Aovvo.reshape(no*nv,nv*no)
    Avoov = Avoov.reshape(nv*no,no*nv)
    Cov = Cov.reshape(no*nv)
    Cvo = Cvo.reshape(nv*no)
    
    Xvo = 1j * np.dot(np.linalg.inv(Aovvo),Cov)
    Xov = 1j * np.dot(np.linalg.inv(Avoov),Cvo)
    Xvo = Xvo.reshape(nv,no)
    Xov = Xov.reshape(no,nv)
    
    X = np.zeros((nmo,nmo),dtype=complex)
    X[:no,no:] = Xov.copy()
    X[no:,:no] = Xvo.copy()
#    print('X: {}'.format(np.linalg.norm(X+X.T.conj())))
    B += einsum('uvpq,pq->uv',A,-1j*X)
#    print(np.linalg.norm(check[:no,no:]))
#    print(np.linalg.norm(check[no:,:no]))

    vv  = einsum('abcd,cd->ab',A[no:,no:,no:,no:],-1j*C[no:,no:])
    vv += B[no:,no:].copy() 
    oo  = einsum('ijkl,kl->ij',A[:no,:no,:no,:no],-1j*C[:no,:no])
    oo += B[:no,:no].copy()
    print('vv: {}'.format(np.linalg.norm(vv+vv.T.conj())))
    print('oo: {}'.format(np.linalg.norm(oo+oo.T.conj())))
    print('vv: {}'.format(np.linalg.norm(vv-C[no:,no:])))
    print('oo: {}'.format(np.linalg.norm(oo-C[:no,:no])))
    return X, 1j*B.T, 1j*C.T 

def compute_energy(d1, d2, eris):
    e  = einsum('pq,qp',eris.h,d1)
    e += 0.25 * einsum('pqrs,rspq',eris.eri,d2)
    return e.real

def ao2mo(Aao, mo_coeff):
    moa, mob = mo_coeff
    Aa = einsum('uv,up,vq->pq',Aao,moa.conj(),moa)
    Ab = einsum('uv,up,vq->pq',Aao,mob.conj(),mob)
    return sort1((Aa,Ab))

def kernel_rt(mf, t, l, U, maxiter=50, step=0.03, omega=1.0):
    no, _, nv, _ = l.shape
    mo0 = mf.mo_coeff.copy()
    U = np.array(U, dtype=complex)
#    print(np.linalg.norm(np.linalg.inv(U)-U.T.conj()))
#    exit()
    mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
    eris = ERIs(mf)
    nao = mf.mol.nao_nr()
    hao_ = np.random.rand(nao,nao) # random time-dependent part
    hao_ += hao_.T.conj()

    X = np.zeros((no+nv,)*2,dtype=complex)
    d1_old, d2 = compute_rdms(t, l)
    d0_old = einsum('qp,vq,up->vu',d1_old,U,U.conj()) # in stationary HF basis
#    check  = einsum('vu,up,vq->qp',d0_old,U,U.conj())
#    print(np.linalg.norm(check-d1_old))
#    exit()
    tr = abs(np.trace(d1_old)-no)
    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        eris.h += ao2mo(hao_, mo_coeff) * math.sin(i*step*omega)
        dt = update_t(t, eris, X) # idt
        dl = update_l(t, l, eris, X) # -idl
        t -= 1j * step * dt
        l += 1j * step * dl
        d1, d2 = compute_rdms(t, l)
        d0 = einsum('qp,vq,up->vu',d1,U,U.conj()) # in stationary HF basis
        dd1, d1_old = d1-d1_old, d1.copy()
        dd0, d0_old = d0-d0_old, d0.copy()
        X, B, C = compute_kappa(d1, d2, eris, dt, dl, t, l)
        C0 = einsum('qp,vq,up->vu',C,U,U.conj()) # in stationary HF basis
        B0 = einsum('qp,vq,up->vu',B,U,U.conj()) # in stationary HF basis
        print(np.linalg.norm(B0 - dd0/step))
        LHS = dd1/step
        tmp  = einsum('rp,qr->qp',X,d1)
        tmp -= einsum('qr,rp->qp',X,d1)
        RHS = C + tmp
        LHS_  = einsum('vu,up,vq->qp',dd0,U,U.conj())/step
#        LHS_  = B.copy()
        dU = np.dot(U, X)
        tmp_  = einsum('vu,up,vq->qp',d0,dU,U.conj())
        tmp_ += einsum('vu,up,vq->qp',d0,U,dU.conj())
        LHS_ += tmp_.copy()
        print(np.linalg.norm(LHS - LHS_))
        U = np.dot(U, scipy.linalg.expm(step*X))
        mo_coeff = np.dot(mo0,U[::2,::2]), np.dot(mo0,U[1::2,1::2])
        tr += abs(np.trace(d1_old)-no)
        print('step: {}, phase: {:.4f}, emo: {}, emo0: {}'.format(
               i, i*step*omega, np.linalg.norm(LHS-RHS), 
               np.linalg.norm(LHS_-RHS)))
    print('check trace: {}'.format(tr))


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

import numpy as np
from pyscf import lib
einsum = lib.einsum

def compute_energy(f0, eri, d, l): # eri in physicists notation
    e  = einsum('pr,rp',f0,d)
    e += 0.5 * einsum('pqrs,rp,sq',eri,d,d)
    e -= 0.5 * einsum('pqsr,rp,sq',eri,d,d)
    e += 0.5 * einsum('pqrs,rspq',eri,l)
    return e

def energy(f0, eri, t1, t2):
    aov, boo, bvv = compute_irred(t1, t2, order=4)
    d = propagate1(aov, boo, bvv)
    l = propagate2(t2, d, maxiter=500)
    return compute_energy(f0, eri, d, l)

def sort1(tup):
    a, b = tup
    na0, na1 = a.shape
    nb0, nb1 = b.shape
    out = np.zeros((na0+nb0,na1+nb1))
    out[ ::2, ::2] = a.copy()
    out[1::2,1::2] = b.copy()
    return out

def sort2(tup, anti):
    aa, ab, bb = tup
    na0, na1, na2, na3 = aa.shape
    nb0, nb1, nb2, nb3 = bb.shape
    out = np.zeros((na0+nb0,na1+nb1,na2+nb2,na3+nb3))
    out[ ::2, ::2, ::2, ::2] = aa.copy() 
    out[1::2,1::2,1::2,1::2] = bb.copy() 
    out[ ::2,1::2, ::2,1::2] = ab.copy()
    out[1::2, ::2,1::2, ::2] = ab.transpose(1,0,3,2).copy()
    if anti:
        out[ ::2,1::2,1::2, ::2] = - ab.transpose(0,1,3,2).copy()
        out[1::2, ::2, ::2,1::2] = - ab.transpose(1,0,2,3).copy()
    return out

def compute_irred(t1, t2, order):
    no, _, nv, _ = t2.shape
    bvv = np.zeros((nv,)*2)
    boo = np.zeros((no,)*2)
    aov = t1.copy()

    if order >= 2:
        Fvv =   0.5 * einsum('klad,klbd->ab',t2,t2)
        Foo = - 0.5 * einsum('kicd,kjcd->ij',t2,t2)

        aov += einsum('ijab,jb->ia',t2,t1)
        bvv += Fvv.copy()
        boo += Foo.copy()

    if order >= 3:
        voov = einsum('ilae,jlbe->ajib',t2,t2)
        aov += einsum('ajib,jb->ia',voov,t1)

    if order >= 4:
        vvvv = 0.5 * einsum('klab,klcd->abcd',t2,t2)
        oooo = 0.5 * einsum('ijcd,klcd->ijkl',t2,t2)

        tmp  = 0.0
        tmp -= einsum('kb,kc,jc->bj',t1,t1,t1)
        tmp -= einsum('jc,bc->bj',t1,Fvv)
        tmp += einsum('kb,jk->bj',t1,Foo)
        aov += einsum('ijab,bj->ia',t2,tmp)
        aov += einsum('ajib,jkbc,kc->ia',voov,t2,t1)
        aov -= einsum('ajkb,jibc,kc->ia',voov,t2,t1)
        aov += 0.5 * einsum('abcd,ijcd,jb->ia',vvvv,t2,t1)

        bvv -= einsum('acbd,kc,kd->ab',vvvv,t1,t1)
        bvv -= einsum('ajib,id,jd->ab',voov,t1,t1)
        bvv += einsum('ij,ajib->ab',Foo,voov)
        bvv -= einsum('fe,aebf->ab',Fvv,vvvv)
        bvv += einsum('ekmc,ikac,imbe->ab',voov,t2,t2)
        bvv += 0.25 * einsum('ijkl,ijae,klbe->ab',oooo,t2,t2)

        boo += einsum('ikjl,kc,lc->ij',oooo,t1,t1)
        boo += einsum('bija,lb,la->ij',voov,t1,t1)
        boo += einsum('ab,ajib->ij',Fvv,voov)
        boo -= einsum('ml,iljm->ij',Foo,oooo)
        boo -= einsum('ekmc,jkbc,imbe->ij',voov,t2,t2)
        boo -= 0.25 * einsum('abcd,mjab,micd->ij',vvvv,t2,t2)

    return aov, boo, bvv 

def propagate1(aov, boo, bvv):
    no, nv = aov.shape
    nmo = no + nv

    sigma = np.zeros((nmo,)*2)
    sigma[:no,:no] = boo.copy()
    sigma[no:,no:] = bvv.copy()
    sigma[:no,no:] = aov.copy()
    sigma[no:,:no] = aov.T.copy() 

    g = np.zeros((nmo,)*2)
    g[no:,no:] = np.eye(nv)
    g[:no,:no] = - np.eye(no)

    d = np.dot(sigma, np.linalg.inv(np.eye(nmo) + np.dot(g,sigma)))
    return d

def propagate2(t2, d, maxiter=100, thresh=1e-8):
    no, _, nv, _ = t2.shape
    nmo = no + nv
    N = np.zeros((nmo,)*4)
    N[:no,:no,no:,no:] = t2.copy()
    N[no:,no:,:no,:no] = t2.transpose(2,3,0,1).copy()

    g = - d.copy()
    g[no:,no:] += np.eye(nv)
    g[:no,:no] -= np.eye(no)

    def _L(G12,G,g):
        tmp = einsum('pquv,ux,vy->pqxy',G12,g,g)
        return 0.5 * einsum('pqxy,xyrs->pqrs',tmp,G)
    def _R(G13,G,g):
        tmp = einsum('purv,vx,yu->pyrx',G13,g,g)
        return - einsum('pyrx,xqys->pqrs',tmp,G)
    
    conv = False
    L = N.copy()
    R = N.copy()
    G = N.copy()
    for i in range(maxiter):
        tmpR = _R(R,G,g)

        L_new = N + tmpR - tmpR.transpose(0,1,3,2)
        R_new = N + _L(L,G,g) - tmpR.transpose(0,1,3,2)
        G_new = L_new + _L(L_new,G,g)
        dnorm = np.linalg.norm(G_new - G)
        L, R, G = L_new.copy(), R_new.copy(), G_new.copy()
#        print('iter: {}, dnorm: {}'.format(i, dnorm))
        if dnorm < thresh:
            conv = True
            break
    if not conv:
        print('Propagation of Gamma2 did not converge!')
    l = einsum('uvxy,xr,ys->uvrs',G,g,g) 
    l = einsum('uvrs,up,vq->pqrs',l,g,g) 
    return l


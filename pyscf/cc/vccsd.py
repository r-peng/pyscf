import numpy as np
import math
from pyscf import lib, fci
from pyscf.fci import cistring
einsum = lib.einsum

def compute_energy_(f0, eri, d, l): # plain eri in physicists notation
    e  = einsum('pr,rp',f0,d)
    e += 0.5 * einsum('pqrs,rp,sq',eri,d,d)
    e -= 0.5 * einsum('pqsr,rp,sq',eri,d,d)
    e += 0.5 * einsum('pqrs,rspq',eri,l)
    return e

def compute_energy(f0, eri, d, l): # anti-symm eri in physicists notation
    e  = einsum('pr,rp',f0,d)
    e += 0.5 * einsum('pqrs,rp,sq',eri,d,d)
    e += 0.25 * einsum('pqrs,rspq',eri,l)
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

def compute_irred_(t1, t2, order):
    no, nv = t1.shape
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

def compute_irred(t1, t2, order):
    no, nv = t1.shape
    bvv = np.zeros((nv,)*2)
    boo = np.zeros((no,)*2)
    aov = t1.copy()

    if order >= 2:
        fvv =   0.5 * einsum('klad,klbd->ab',t2,t2)
        foo = - 0.5 * einsum('kicd,kjcd->ij',t2,t2)

        aov += einsum('ijab,jb->ia',t2,t1)
        bvv += fvv.copy()
        boo += foo.copy()

    if order >= 3:
        voov = einsum('ilae,jlbe->ajib',t2,t2)
        aov += einsum('ajib,jb->ia',voov,t1)

    if order >= 4:
        Fvv = fvv + einsum('ia,ib->ab',t1,t1)
        Foo = foo - einsum('ia,ja->ij',t1,t1)
        vvvv = 0.5 * einsum('klab,klcd->abcd',t2,t2)
        oooo = 0.5 * einsum('ijcd,klcd->ijkl',t2,t2)
        Roovv = einsum('ajib,jkbc->ikac',voov,t2)
        Loovv = 0.5 * einsum('ijkl,klab->ijab',oooo,t2) 
        Toovv = Roovv + 0.5 * Loovv 

        tmp  = 0.0
        tmp -= einsum('jc,bc->bj',t1,Fvv)
        tmp += einsum('kb,jk->bj',t1,foo)
        aov += einsum('ijab,bj->ia',t2,tmp)
        aov += einsum('ikac,kc->ia',Roovv,t1)
        aov -= einsum('kiac,kc->ia',Roovv,t1)
        aov += einsum('ijab,jb->ia',Loovv,t1)

        bvv += einsum('ij,ajib->ab',Foo,voov)
        bvv -= einsum('fe,aebf->ab',Fvv,vvvv)
        bvv += einsum('imae,imbe->ab',Toovv,t2)

        boo += einsum('ab,ajib->ij',Fvv,voov)
        boo -= einsum('ml,iljm->ij',Foo,oooo)
        boo -= einsum('jmbe,imbe->ij',Toovv,t2)
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

def compute_ci(t1, t2, nfc, nfv):
    noa, nob, nva, nvb = t2[1].shape
    ne = noa + nfc[0], nob + nfc[1]
    nmo = ne[0] + nva + nfv[0], ne[1] + nvb + nfv[1]
    Na = cistring.num_strings(nmo[0],ne[0])
    Nb = cistring.num_strings(nmo[0],ne[1])
    ci = np.zeros((Na,Nb))
    ci[0,0] = 1.0
    
    Sa, Taa = ci_imt(nmo[0],nfc[0],t1[0],t2[0])
    Sb, Tbb = ci_imt(nmo[1],nfc[1],t1[1],t2[2])
    temp = einsum('KIia,ijab->KIjb',Sa,t2[1]) 
    def T(c):
        c_  = np.dot(Taa,c)
        c_ += np.dot(Tbb,c.T).T
        c_ += einsum('KIjb,LJjb,IJ->KL',temp,Sb,c)
        return c_
    out = ci.copy() 
    for n in range(1, sum(ne)+1):
        ci = T(ci)
        out += ci/math.factorial(n)
    return out/np.linalg.norm(out)

def ci_imt(nmo,nfc,t1,t2):
    no, nv = t1.shape
    ne = no + nfc
    N = cistring.num_strings(nmo,ne)
    S = np.zeros((N, N, no, nv))
    T = np.zeros((N, N))
    for I in range(N): 
        strI = cistring.addr2str(nmo,ne,I) 
        for i in range(no):
            des1 = i+nfc
            h1 = 1 << des1
            if strI & h1 != 0:
                for a in range(nv):
                    cre1 = a + ne
                    p1 = 1 << cre1
                    if strI & p1 == 0:
                        str1 = strI ^ h1 | p1
                        K = cistring.str2addr(nmo,ne,str1)
                        sgn1 = cistring.cre_des_sign(cre1,des1,strI)
                        S[K,I,i,a] += sgn1
                        T[K,I] += t1[i,a]*sgn1
                        for j in range(i):
                            des2 = j+nfc
                            h2 = 1 << des2
                            if strI & h2 != 0:
                                for b in range(a):
                                    cre2 = b + ne
                                    p2 = 1 << cre2
                                    if strI & p2 == 0:
                                        str2 = str1 ^ h2 | p2
                                        K = cistring.str2addr(nmo,ne,str2)
                                        sgn2 = cistring.cre_des_sign(cre2,des2,str1)
                                        T[K,I] += t2[i,j,a,b]*sgn1*sgn2
    return S, T 

import numpy as np
from pyscf import lib
einsum = lib.einsum

def compute_energy(f0, eri, d, l): # eri in physicists notation
    e  = einsum('pr,rp',f0,d)
    e += 0.5 * einsum('pqrs,rp,sq',eri,d,d)
    e -= 0.5 * einsum('pqsr,rp,sq',eri,d,d)
    e += 0.5 * einsum('pqrs,rspq',eri,l)
    return e

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
#    print('d symm: {}'.format(np.linalg.norm(d-d.T)))
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
#        print('G12 symm:{},{},{},{}'.format(
#            np.linalg.norm(G12+G12.transpose(1,0,2,3)),
#            np.linalg.norm(G12+G12.transpose(0,1,3,2)),
#            np.linalg.norm(G12-G12.transpose(1,0,3,2)),
#            np.linalg.norm(G12-G12.transpose(2,3,0,1))))
#        print('G13 symm:{},{}'.format(
#                np.linalg.norm(G13-G13.transpose(1,0,3,2)),
#                np.linalg.norm(G13-G13.transpose(2,3,0,1))))
#        print('G14 symm:{},{}'.format(
#                np.linalg.norm(G14-G14.transpose(1,0,3,2)),
#                np.linalg.norm(G14-G14.transpose(2,3,0,1))))
#        print('G symm:{},{},{},{}'.format(
#            np.linalg.norm(G+G.transpose(1,0,2,3)),
#            np.linalg.norm(G+G.transpose(0,1,3,2)),
#            np.linalg.norm(G-G.transpose(1,0,3,2)),
#            np.linalg.norm(G-G.transpose(2,3,0,1))))
#        print('G13/14 symm:{}'.format(np.linalg.norm(G13+G14.transpose(0,1,3,2))))
        if dnorm < thresh:
            conv = True
            break
    if not conv:
        print('Propagation of Gamma2 did not converge!')
#    print('G symm:\n{}\n{}\n{}\n{}'.format(
#        np.linalg.norm(G+G.transpose(1,0,2,3)),
#        np.linalg.norm(G+G.transpose(0,1,3,2)),
#        np.linalg.norm(G-G.transpose(1,0,3,2)),
#        np.linalg.norm(G-G.transpose(2,3,0,1))))
    l = einsum('uvxy,xr,ys->uvrs',G,g,g) 
    l = einsum('uvrs,up,vq->pqrs',l,g,g) 
    return l

if __name__ == '__main__':
    from pyscf import gto, scf, ao2mo, fci, cc

    mol = gto.Mole()
    mol.atom = [['B',(0, 0, 0)], ['H',(0, 3.0, 0)]] 
    mol.basis = 'sto-3g'
    mol.symmetry = False
    mol.build()
    nmo = mol.nao_nr()
    noa, nob = mol.nelec
    no = noa + nob
    
    mf = scf.RHF(mol)
    mf.kernel()
    eri = mf.mol.intor('int2e_sph', aosym='s8')
    eri = ao2mo.incore.full(eri,mf.mo_coeff)
    eri = ao2mo.restore(1,eri,nmo)
    eri = eri.transpose(0,2,1,3)
    f0 = np.diag(mf.mo_energy)

    cisolver = fci.FCI(mol, mf.mo_coeff)
    cisolver.kernel()   

    (da, db), (laa, lab, lbb) = cisolver.make_rdm12s(cisolver.ci,nmo,(noa,nob))
    laa  = laa.transpose(0,2,1,3)
    laa -= einsum('pr,qs->pqrs',da,da) 
    laa += einsum('ps,qr->pqrs',da,da)  
    lbb  = lbb.transpose(0,2,1,3)
    lbb -= einsum('pr,qs->pqrs',db,db) 
    lbb += einsum('ps,qr->pqrs',db,db)  
    lab  = lab.transpose(0,2,1,3)
    lab -= einsum('pr,qs->pqrs',da,db) 
    da[:noa,:noa] -= np.eye(noa)
    db[:nob,:nob] -= np.eye(nob)

    f0 = sort1((f0, f0))
    d = sort1((da, db))
    eri = sort2((eri, eri, eri), anti=False)
    l = sort2((laa, lab, lbb), anti=True)
    e = compute_energy(f0, eri, d, l)
    print('FCI energy check: {}'.format(cisolver.eci-mf.e_tot-e))

    mycc = cc.UCCSD(mf.to_uhf())
    emp2, t1, t2 = mycc.init_amps() 
    t1 = sort1(t1)
    t2 = sort2(t2, anti=True)

    _, doo, dvv = compute_irred(t1, t2, order=2)
    d = np.zeros((nmo*2,)*2)
    d[:no,:no] = doo.copy()
    d[no:,no:] = dvv.copy()
    l = np.zeros((nmo*2,)*4)
    l[:no,:no,no:,no:] = t2.copy()
    l[no:,no:,:no,:no] = t2.transpose(2,3,0,1)
    e  = einsum('pr,rp',f0,d)
    e += 0.5 * einsum('pqrs,rspq',eri,l)
    print('MP2 energy check: {}'.format(emp2-e))

    mycc.kernel()
    t1 = sort1(mycc.t1)
    t2 = sort2(mycc.t2, anti=True)
    aov, boo, bvv = compute_irred(t1, t2, order=4)
    check  = t1.copy()
    check += einsum('ijab,jb->ia',t2,t1)
    check += einsum('ijab,jkbc,kc->ia',t2,t2,t1)
    check -= einsum('ijab,kb,kc,jc->ia',t2,t1,t1,t1)
    check -= 0.5 * einsum('ijab,klbd,klcd,jc->ia',t2,t2,t2,t1)
    check -= 0.5 * einsum('ijab,jlcd,klcd,kb->ia',t2,t2,t2,t1)
    check += einsum('ijab,jkbc,klcd,ld->ia',t2,t2,t2,t1)
    check -= einsum('ijdb,jkbc,klca,ld->ia',t2,t2,t2,t1)
    check += 0.25 * einsum('klab,klcd,ijcd,jb->ia',t2,t2,t2,t1)
    print('check ov: {}'.format(np.linalg.norm(aov-check)))
    check  = 0.0
    check += 0.5 * einsum('klad,klbd->ab',t2,t2)
    check -= 0.5 * einsum('ijac,ijbd,kc,kd->ab',t2,t2,t1,t1)
    check -= einsum('ikac,jkbc,id,jd->ab',t2,t2,t1,t1)
    check -= 0.5 * einsum('kjcd,kicd,ilae,jlbe->ab',t2,t2,t2,t2)
    check -= 0.25 * einsum('klec,klfc,ijaf,ijbe->ab',t2,t2,t2,t2)
    check += einsum('klcd,lmde,ikac,imbe->ab',t2,t2,t2,t2)
    check += 0.125 * einsum('ijcd,klcd,ijae,klbe->ab',t2,t2,t2,t2) 
    print('check vv: {}'.format(np.linalg.norm(bvv-check)))
#    exit()
    check  = 0.0
    check -= 0.5 * einsum('kjcd,kicd->ij',t2,t2)
    check += 0.5 * einsum('ikab,jlab,kc,lc->ij',t2,t2,t1,t1)
    check += einsum('ikac,jkbc,lb,la->ij',t2,t2,t1,t1)
    check += 0.5 * einsum('klad,klbd,imae,jmbe->ij',t2,t2,t2,t2)
    check += 0.25 * einsum('kmcd,klcd,mjab,liab->ij',t2,t2,t2,t2)
    check -= einsum('klcd,lmde,jkbc,imbe->ij',t2,t2,t2,t2)
    check -= 0.125 * einsum('klab,klcd,mjab,micd->ij',t2,t2,t2,t2)
    print('check oo: {}'.format(np.linalg.norm(boo-check)))

    fov = f0[:no,no:]
    oovv = eri[:no,:no,no:,no:]
    e  = einsum('ia,ia',fov,t1)
    e += 0.5 * einsum('ijab,ijab',oovv,t2)
    e += 0.5 * einsum('ijab,ia,jb',oovv,t1,t1)
    e -= 0.5 * einsum('jiab,ia,jb',oovv,t1,t1)
    print('CC energy check: {}'.format(mycc.e_corr-e))

    

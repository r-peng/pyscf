import numpy as np
import scipy
from pyscf import lib, ao2mo
einsum = lib.einsum

def update_amps(t, l, eris):
    taa, tab, tbb = t
    laa, lab, lbb = l
    fa, fb = eris.f
    eri_aa, eri_ab, eri_bb = eris.eri
    _, _, noa, nob = tab.shape

    Foo  = fa[:noa,:noa].copy()
    Foo += 0.5 * einsum('klcd,cdjl->kj',eri_aa[:noa,:noa,noa:,noa:],taa)
    Foo +=       einsum('kLcD,cDjL->kj',eri_ab[:noa,:noa,noa:,noa:],tab)
    Fvv  = fa[noa:,noa:].copy()
    Fvv -= 0.5 * einsum('klcd,bdkl->bc',eri_aa[:noa,:noa,noa:,noa:],taa)
    Fvv -=       einsum('kLcD,bDkL->bc',eri_ab[:noa,:noa,noa:,noa:],tab)
    FOO = Foo.copy()
    FVV = Fvv.copy()

    dtaa  = eri_aa[noa:,noa:,:noa,:noa].copy()
    dtaa += einsum('bc,acij->abij',Fvv,taa)
    dtaa += einsum('ac,cbij->abij',Fvv,taa)
    dtaa -= einsum('kj,abik->abij',Foo,taa)
    dtaa -= einsum('ki,abkj->abij',Foo,taa)

    dtab  = eri_ab[noa:,nob:,:noa,:nob].copy()
    dtab += einsum('BC,aCiJ->aBiJ',FVV,tab)
    dtab += einsum('ac,cBiJ->aBiJ',Fvv,tab)
    dtab -= einsum('KJ,aBiK->aBiJ',FOO,tab)
    dtab -= einsum('ki,aBkJ->aBiJ',Foo,tab)

    dlaa  = eri_aa[:noa,:noa,noa:,noa:].copy()
    dlaa += einsum('cb,ijac->ijab',Fvv,laa)
    dlaa += einsum('ca,ijcb->ijab',Fvv,laa)
    dlaa -= einsum('jk,ikab->ijab',Foo,laa)
    dlaa -= einsum('ik,kjab->ijab',Foo,laa)

    dlab  = eri_ab[:noa,:nob,noa:,nob:].copy()
    dlab += einsum('CB,iJaC->iJaB',FVV,lab)
    dlab += einsum('ca,iJcB->iJaB',Fvv,lab)
    dlab -= einsum('JK,iKaB->iJaB',FOO,lab)
    dlab -= einsum('ik,kJaB->iJaB',Foo,lab)

    loooo  = eri_aa[:noa,:noa,:noa,:noa].copy()
    loooo += 0.5 * einsum('klcd,cdij->klij',eri_aa[:noa,:noa,noa:,noa:],taa)
    loOoO  = eri_ab[:noa,:nob,:noa,:nob].copy()
    loOoO +=       einsum('kLcD,cDiJ->kLiJ',eri_ab[:noa,:nob,noa:,nob:],tab)
    lvvvv  = eri_aa[noa:,noa:,noa:,noa:].copy()
    lvvvv += 0.5 * einsum('klab,cdkl->cdab',eri_aa[:noa,:noa,noa:,noa:],taa)
    lvVvV  = eri_ab[noa:,nob:,noa:,nob:].copy()
    lvVvV +=       einsum('kLaB,cDkL->cDaB',eri_ab[:noa,:nob,noa:,nob:],tab)

    dtaa += 0.5 * einsum('abcd,cdij->abij',eri_aa[noa:,noa:,noa:,noa:],taa)
    dtab +=       einsum('aBcD,cDiJ->aBiJ',eri_ab[noa:,nob:,noa:,nob:],tab)
    dtaa += 0.5 * einsum('klij,abkl->abij',loooo,taa)
    dtab +=       einsum('kLiJ,aBkL->aBiJ',loOoO,tab)

    dlaa += 0.5 * einsum('cdab,ijcd->ijab',lvvvv,laa)
    dlab +=       einsum('cDaB,iJcD->iJaB',lvVvV,lab)
    dlaa += 0.5 * einsum('ijkl,klab->ijab',loooo,laa)
    dlab +=       einsum('iJkL,kLaB->iJaB',loOoO,lab)

    tmp  = einsum('bkjc,acik->abij',eri_aa[noa:,:noa,:noa,noa:],taa)
    tmp += einsum('bKjC,aCiK->abij',eri_ab[noa:,:nob,:noa,nob:],tab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dtaa += tmp.copy()
    dtab += einsum('kBcJ,acik->aBiJ',eri_ab[:noa,nob:,noa:,:nob],taa)
    dtab += einsum('KBCJ,aCiK->aBiJ',eri_bb[:nob,nob:,nob:,:nob],tab)
    dtab -= einsum('kBiC,aCkJ->aBiJ',eri_ab[:noa,nob:,:noa,nob:],tab)
    dtab -= einsum('aKcJ,cBiK->aBiJ',eri_ab[noa:,:nob,noa:,:nob],tab)
    dtab += einsum('akic,cBkJ->aBiJ',eri_aa[noa:,:noa,:noa,noa:],tab)
    dtab += einsum('aKiC,BCJK->aBiJ',eri_ab[noa:,:nob,:noa,nob:],tbb)

    tmp  = einsum('jcbk,ikac->ijab',eri_aa[:noa,noa:,noa:,:noa],laa)
    tmp += einsum('jCbK,iKaC->ijab',eri_ab[:noa,nob:,noa:,:nob],lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlaa += tmp.copy()
    dlab += einsum('cJkB,ikac->iJaB',eri_ab[noa:,:nob,:noa,nob:],laa)
    dlab += einsum('CJKB,iKaC->iJaB',eri_bb[nob:,:nob,:nob,nob:],lab)
    dlab -= einsum('cJaK,iKcB->iJaB',eri_ab[noa:,:nob,noa:,:nob],lab)
    dlab -= einsum('iCkB,kJaC->iJaB',eri_ab[:noa,nob:,:noa,nob:],lab)
    dlab += einsum('icak,kJcB->iJaB',eri_aa[:noa,noa:,noa:,:noa],lab)
    dlab += einsum('iCaK,JKBC->iJaB',eri_ab[:noa,nob:,noa:,:nob],lbb)

    rovvo  = einsum('klcd,bdjl->kbcj',eri_aa[:noa,:noa,noa:,noa:],taa)
    rovvo += einsum('kLcD,bDjL->kbcj',eri_ab[:noa,:nob,noa:,nob:],tab)
    rvOoV  = einsum('lKdC,bdjl->bKjC',eri_ab[:noa,:nob,noa:,nob:],taa)
    rvOoV += einsum('KLCD,bDjL->bKjC',eri_bb[:nob,:nob,nob:,nob:],tab)
    roVvO  = einsum('kLcD,BDJL->kBcJ',eri_ab[:noa,:nob,noa:,nob:],tbb)
    roVvO += einsum('klcd,dBlJ->kBcJ',eri_aa[:noa,:noa,noa:,noa:],tab)
    rOVVO  = rovvo.copy()
    roVoV  = einsum('kLdC,dBiL->kBiC',eri_ab[:noa,:nob,noa:,nob:],tab)
    rvOvO  = einsum('lKcD,bDlI->bKcI',eri_ab[:noa,:nob,noa:,nob:],tab)

    tmp  = einsum('kbcj,acik->abij',rovvo,taa)
    tmp += einsum('bKjC,aCiK->abij',rvOoV,tab)
    tmp -= tmp.transpose(0,1,3,2)
    dtaa += tmp.copy()
    dtab += einsum('kBcJ,acik->aBiJ',roVvO,taa)
    dtab += einsum('KBCJ,aCiK->aBiJ',rOVVO,tab)
    dtab += einsum('kBiC,aCkJ->aBiJ',roVoV,tab)

    tmp  = einsum('jcbk,ikac->ijab',rovvo,laa)
    tmp += einsum('jCbK,iKaC->ijab',roVvO,lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlaa += tmp.copy()

    dlab += einsum('cJkB,ikac->iJaB',rvOoV,laa)
    dlab += einsum('JCBK,iKaC->iJaB',rOVVO,lab)
    dlab += einsum('iCkB,kJaC->iJaB',roVoV,lab)
    dlab += einsum('cJaK,iKcB->iJaB',rvOvO,lab)
    dlab += einsum('icak,kJcB->iJaB',rovvo,lab)
    dlab += einsum('iCaK,JKBC->iJaB',roVvO,lbb)

    Foo  = 0.5 * einsum('ilcd,cdkl->ik',laa,taa)
    Foo +=       einsum('iLcD,cDkL->ik',lab,tab)
    Fvv  = 0.5 * einsum('klad,cdkl->ca',laa,taa)
    Fvv +=       einsum('kLaD,cDkL->ca',lab,tab)
    FOO = Foo.copy()
    FVV = Fvv.copy()

    dlaa -= einsum('ik,kjab->ijab',Foo,eri_aa[:noa,:noa,noa:,noa:])
    dlaa -= einsum('jk,ikab->ijab',Foo,eri_aa[:noa,:noa,noa:,noa:])
    dlaa -= einsum('ca,ijcb->ijab',Fvv,eri_aa[:noa,:noa,noa:,noa:])
    dlaa -= einsum('cb,ijac->ijab',Fvv,eri_aa[:noa,:noa,noa:,noa:])

    dlab -= einsum('ik,kJaB->iJaB',Foo,eri_ab[:noa,:nob,noa:,nob:])
    dlab -= einsum('JK,iKaB->iJaB',FOO,eri_ab[:noa,:nob,noa:,nob:])
    dlab -= einsum('ca,iJcB->iJaB',Fvv,eri_ab[:noa,:nob,noa:,nob:])
    dlab -= einsum('CB,iJaC->iJaB',FVV,eri_ab[:noa,:nob,noa:,nob:])

    dtbb = dtaa.copy()
    dlbb = dlaa.copy()
    return (dtaa, dtab, dtbb), (dlaa, dlab, dlbb)

def compute_gamma(t, l): # normal ordered, asymmetric
    taa, tab, tbb = t
    laa, lab, lbb = l
    dvv  = 0.5 * einsum('ikac,bcik->ba',laa,taa)
    dvv +=       einsum('iKaC,bCiK->ba',lab,tab)
    doo  = 0.5 * einsum('jkac,acik->ji',laa,taa)
    doo +=       einsum('jKaC,aCiK->ji',lab,tab)
    doo *= - 1.0
    dVV = dvv.copy()
    dOO = doo.copy()

    dovvo  = einsum('jkbc,acik->jabi',laa,taa)
    dovvo += einsum('jKbC,aCiK->jabi',lab,tab)
    doVvO  = einsum('jkbc,cAkI->jAbI',laa,tab)
    doVvO += einsum('jKbC,ACIK->jAbI',lab,tbb)
    dvOoV  = einsum('kJcB,acik->aJiB',lab,taa)
    dvOoV += einsum('JKBC,aCiK->aJiB',lbb,tab)
    doVoV  = - einsum('jKcB,cAiK->jAiB',lab,tab)
    dvOvO  = - einsum('kJbC,aCkI->aJbI',lab,tab) 
    dOVVO  = dovvo.copy()

    dvvvv = 0.5 * einsum('ijab,cdij->cdab',laa,taa)
    dvVvV =       einsum('iJaB,cDiJ->cDaB',lab,tab)
    doooo = 0.5 * einsum('klab,abij->klij',laa,taa)
    doOoO =       einsum('kLaB,aBiJ->kLiJ',lab,tab)
    dVVVV = dvvvv.copy()
    dOOOO = doooo.copy()

    dvvoo = taa.copy()
    tmp  = einsum('ladi,bdjl->abij',dovvo,taa)
    tmp += einsum('aLiD,bDjL->abij',dvOoV,tab)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo += tmp.copy()
    dvvoo -= einsum('ac,cbij->abij',dvv,taa)
    dvvoo -= einsum('bc,acij->abij',dvv,taa)
    dvvoo += einsum('ki,abkj->abij',doo,taa)
    dvvoo += einsum('kj,abik->abij',doo,taa)
    dvvoo += 0.5 * einsum('klij,abkl->abij',doooo,taa)

    dvVoO  = tab.copy()
    dvVoO += einsum('ladi,dBlJ->aBiJ',dovvo,tab)
    dvVoO += einsum('aLiD,BDJL->aBiJ',dvOoV,tbb)
    dvVoO -= einsum('aLdJ,dBiL->aBiJ',dvOvO,tab)
    dvVoO -= einsum('ac,cBiJ->aBiJ',dvv,tab)
    dvVoO -= einsum('BC,aCiJ->aBiJ',dVV,tab)
    dvVoO += einsum('ki,aBkJ->aBiJ',doo,tab)
    dvVoO += einsum('KJ,aBiK->aBiJ',dOO,tab)
    dvVoO += einsum('kLiJ,aBkL->aBiJ',doOoO,tab) 
    dVVOO = dvvoo.copy()

    doo = doo, dOO
    dvv = dvv, dVV
    doooo = doooo, doOoO, dOOOO
    dvvvv = dvvvv, dvVvV, dVVVV
    doovv = l
    dvvoo = dvvoo, dvVoO, dVVOO
    dovvo = dovvo, doVvO, dvOoV, doVoV, dvOvO, dOVVO 
    return doo, dvv, doooo, doovv, dvvoo, dovvo, dvvvv 

def compute_rdms(t, l, normal=False, symm=True):
    doo, dvv, doooo, doovv, dvvoo, dovvo, dvvvv = compute_gamma(t, l)
    doo, dOO = doo
    dvv, dVV = dvv
    doooo, doOoO, dOOOO = doooo
    dvvvv, dvVvV, dVVVV = dvvvv
    doovv, doOvV, dOOVV = doovv
    dvvoo, dvVoO, dVVOO = dvvoo
    dovvo, doVvO, dvOoV, doVoV, dvOvO, dOVVO = dovvo 

    noa, nob, nva, nvb = doOvV.shape
    nmoa, nmob = noa + nva, nob + nvb
    if not normal:
        doooo += einsum('ki,lj->klij',np.eye(noa),doo)
        doooo += einsum('lj,ki->klij',np.eye(noa),doo)
        doooo -= einsum('li,kj->klij',np.eye(noa),doo)
        doooo -= einsum('kj,li->klij',np.eye(noa),doo)
        doooo += einsum('ki,lj->klij',np.eye(noa),np.eye(noa))
        doooo -= einsum('li,kj->klij',np.eye(noa),np.eye(noa))

        doOoO += einsum('ki,lj->klij',np.eye(noa),dOO)
        doOoO += einsum('lj,ki->klij',np.eye(nob),doo)
        doOoO += einsum('ki,lj->klij',np.eye(noa),np.eye(nob))

        dovvo -= einsum('ji,ab->jabi',np.eye(noa),dvv)
        doVoV += einsum('ji,AB->jAiB',np.eye(noa),dVV)
        dvOvO += einsum('JI,ab->aJbI',np.eye(nob),dvv)

        doo += np.eye(noa)

    da = np.zeros((nmoa,nmoa),dtype=complex)
    da[:noa,:noa] = doo.copy()
    da[noa:,noa:] = dvv.copy()
    daa = np.zeros((nmoa,nmoa,nmoa,nmoa),dtype=complex)
    daa[:noa,:noa,:noa,:noa] = doooo.copy()
    daa[:noa,:noa,noa:,noa:] = doovv.copy()
    daa[noa:,noa:,:noa,:noa] = dvvoo.copy()
    daa[:noa,noa:,noa:,:noa] = dovvo.copy()
    daa[noa:,:noa,:noa,noa:] = dovvo.transpose(1,0,3,2)
    daa[:noa,noa:,:noa,noa:] = - dovvo.transpose(0,1,3,2)
    daa[noa:,:noa,noa:,:noa] = - dovvo.transpose(1,0,2,3)
    daa[noa:,noa:,noa:,noa:] = dvvvv.copy()
    dab = np.zeros((nmoa,nmob,nmoa,nmob),dtype=complex)
    dab[:noa,:nob,:noa,:nob] = doOoO.copy()
    dab[:noa,:nob,noa:,nob:] = doOvV.copy()
    dab[noa:,nob:,:noa,:nob] = dvVoO.copy()
    dab[:noa,nob:,noa:,:nob] = doVvO.copy()
    dab[noa:,:nob,:noa,nob:] = dvOoV.copy()
    dab[:noa,nob:,:noa,nob:] = doVoV.copy()
    dab[noa:,:nob,noa:,:nob] = dvOvO.copy()
    dab[noa:,nob:,noa:,nob:] = dvVvV.copy()
 
    if symm:
        da = 0.5 * (da + da.T.conj())
        daa = 0.5 * (daa + daa.transpose(2,3,0,1).conj())
        dab = 0.5 * (dab + dab.transpose(2,3,0,1).conj())

    db = da.copy()
    dbb = daa.copy()
    return (da, db), (daa, dab, dbb)

def compute_kappa_intermediates(d1, d2, eris, no):
    nv = d1.shape[0] - no
    h = eris.h
    eri = eris.eri    

    Cov  = einsum('ba,aj->jb',d1[no:,no:],h[no:,:no]) 
    Cov -= einsum('ij,bi->jb',d1[:no,:no],h[no:,:no])
    Cov += 0.5 * einsum('pqjs,bspq->jb',eri[:,:,:no,:],d2[no:,:,:,:])
    Cov -= 0.5 * einsum('bqrs,rsjq->jb',eri[no:,:,:,:],d2[:,:,:no,:])

    Aovvo  = einsum('ba,ij->jbai',np.eye(nv),d1[:no,:no])
    Aovvo -= einsum('ij,ba->jbai',np.eye(no),d1[no:,no:])
    return Aovvo, Cov

def compute_kappa(d1, d2, eris, no):
    Aovvo, Cov = compute_kappa_intermediates(d1, d2, eris, no)
    nv = d1.shape[0] - no
    Aovvo = Aovvo.reshape(no*nv,no*nv)
    Cov = Cov.reshape(no*nv)
    kappa = np.dot(np.linalg.inv(Aovvo),Cov)
    kappa = kappa.reshape(nv,no)
    kappa = np.block([[np.zeros((no,no)),-kappa.T.conj()],
                      [kappa, np.zeros((nv,nv))]])
    return kappa

def compute_energy(d1, d2, eris):
    h = eris.h
    eri = eris.eri    
    e  = einsum('pq,qp',h,d1)
    e += 0.25 * einsum('pqrs,rspq',eri,d2)
    return e.real

def kernel_it1(mf, maxiter=1000, step=0.03, thresh=1e-6):
    no = sum(mf.mol.nelec)

def kernel_it(mf, maxiter=1000, step=0.03, thresh=1e-8):
    no = sum(mf.mol.nelec)
    eris = ERIs(mf)
    mo_coeff = mf.mo_coeff.copy()
    eo = np.diag(eris.f[:no,:no])
    ev = np.diag(eris.f[no:,no:])
    eia = lib.direct_sum('i-a->ia', eo, ev)
    eabij = lib.direct_sum('ia+jb->abij', eia, eia)
    t = eris.eri[no:,no:,:no,:no]/eabij
    l = t.transpose(2,3,0,1).copy()
    d1, d2 = compute_rdms(t, l)
    e = compute_energy(d1, d2, eris)

    converged = False
    for i in range(maxiter):
        kappa = compute_kappa(d1, d2, eris, no)
        U = scipy.linalg.expm(step*kappa[::2,::2]) # U = U_{old,new}
        mo_coeff = np.dot(mo_coeff, U)
        eris.ao2mo(mo_coeff)
        dt = update_t(t, eris)
        dl = update_l(t, l, eris)
        t -= step * dt
        l -= step * dl
        d1, d2 = compute_rdms(t, l)
        e_new = compute_energy(d1, d2, eris)
        de, e = e_new - e, e_new
        dnormk = np.linalg.norm(kappa)
        dnormt = np.linalg.norm(dt)
        dnorml = np.linalg.norm(dl)
        print('iter: {}, dk: {}, dt: {}, dl: {}, de: {}, energy: {}'.format(
              i, dnormk, dnormt, dnorml, de, e))
        if dnormk < thresh:
            converged = True
            break
    return t, l, mo_coeff, e 

class ERIs:
    def __init__(self, mf):
        self.hao = mf.get_hcore().astype(complex)
        self.fao = mf.get_fock().astype(complex)
        self.eri_ao = mf.mol.intor('int2e_sph').astype(complex)
        self.ao2mo(mf.mo_coeff)

    def ao2mo(self, mo_coeff):
        nmo = mo_coeff.shape[0]
    
        h = einsum('uv,up,vq->pq',self.hao,mo_coeff.conj(),mo_coeff)
        self.h = h, h
    
        f = einsum('uv,up,vq->pq',self.fao,mo_coeff.conj(),mo_coeff)
        self.f = f, f
    
        eri = einsum('uvxy,up,vr->prxy',self.eri_ao,mo_coeff.conj(),mo_coeff)
        eri = einsum('prxy,xq,ys->prqs',eri,mo_coeff.conj(),mo_coeff)
        eri = eri.transpose(0,2,1,3)
        eri_aa = eri - eri.transpose(0,1,3,2)
        self.eri = eri_aa, eri, eri_aa

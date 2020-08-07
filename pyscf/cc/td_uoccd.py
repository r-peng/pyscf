import numpy as np
import scipy
from pyscf import lib, ao2mo
einsum = lib.einsum

def update_amps(t, l, eris):
    taa, tab, tbb = t
    laa, lab, lbb = l
    eri_aa, eri_ab, eri_bb = eris.eri
    _, _, noa, nob = tab.shape
    fa, fb = eris.h[0].copy(), eris.h[1].copy()
    fa += einsum('piqi->pq',eri_aa[:,:noa,:,:noa])
    fa += einsum('pIqI->pq',eri_ab[:,:nob,:,:nob])
    fb += einsum('PIQI->PQ',eri_bb[:,:nob,:,:nob])
    fb += einsum('iPiQ->PQ',eri_ab[:noa,:,:noa,:])

    Foo  = fa[:noa,:noa].copy()
    Foo += 0.5 * einsum('klcd,cdjl->kj',eri_aa[:noa,:noa,noa:,noa:],taa)
    Foo +=       einsum('kLcD,cDjL->kj',eri_ab[:noa,:nob,noa:,nob:],tab)
    Fvv  = fa[noa:,noa:].copy()
    Fvv -= 0.5 * einsum('klcd,bdkl->bc',eri_aa[:noa,:noa,noa:,noa:],taa)
    Fvv -=       einsum('kLcD,bDkL->bc',eri_ab[:noa,:nob,noa:,nob:],tab)

    FOO  = fb[:nob,:nob].copy()
    FOO += 0.5 * einsum('KLCD,CDJL->KJ',eri_bb[:nob,:nob,nob:,nob:],tbb)
    FOO +=       einsum('lKdC,dClJ->KJ',eri_ab[:noa,:nob,noa:,nob:],tab)
    FVV  = fb[nob:,nob:].copy()
    FVV -= 0.5 * einsum('KLCD,BDKL->BC',eri_bb[:nob,:nob,nob:,nob:],tbb)
    FVV -=       einsum('lKdC,dBlK->BC',eri_ab[:noa,:nob,noa:,nob:],tab)

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

    dtbb  = eri_bb[nob:,nob:,:nob,:nob].copy()
    dtbb += einsum('BC,ACIJ->ABIJ',FVV,tbb)
    dtbb += einsum('AC,CBIJ->ABIJ',FVV,tbb)
    dtbb -= einsum('KJ,ABIK->ABIJ',FOO,tbb)
    dtbb -= einsum('KI,ABKJ->ABIJ',FOO,tbb)

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

    dlbb  = eri_bb[:nob,:nob,nob:,nob:].copy()
    dlbb += einsum('CB,IJAC->IJAB',FVV,lbb)
    dlbb += einsum('CA,IJCB->IJAB',FVV,lbb)
    dlbb -= einsum('JK,IKAB->IJAB',FOO,lbb)
    dlbb -= einsum('IK,KJAB->IJAB',FOO,lbb)

    loooo  = eri_aa[:noa,:noa,:noa,:noa].copy()
    loooo += 0.5 * einsum('klcd,cdij->klij',eri_aa[:noa,:noa,noa:,noa:],taa)
    loOoO  = eri_ab[:noa,:nob,:noa,:nob].copy()
    loOoO +=       einsum('kLcD,cDiJ->kLiJ',eri_ab[:noa,:nob,noa:,nob:],tab)
    lOOOO  = eri_bb[:nob,:nob,:nob,:nob].copy()
    lOOOO += 0.5 * einsum('KLCD,CDIJ->KLIJ',eri_bb[:nob,:nob,nob:,nob:],tbb)
    lvvvv  = eri_aa[noa:,noa:,noa:,noa:].copy()
    lvvvv += 0.5 * einsum('klab,cdkl->cdab',eri_aa[:noa,:noa,noa:,noa:],taa)
    lvVvV  = eri_ab[noa:,nob:,noa:,nob:].copy()
    lvVvV +=       einsum('kLaB,cDkL->cDaB',eri_ab[:noa,:nob,noa:,nob:],tab)
    lVVVV  = eri_bb[nob:,nob:,nob:,nob:].copy()
    lVVVV += 0.5 * einsum('KLAB,CDKL->CDAB',eri_bb[:nob,:nob,nob:,nob:],tbb)

    dtaa += 0.5 * einsum('abcd,cdij->abij',eri_aa[noa:,noa:,noa:,noa:],taa)
    dtab +=       einsum('aBcD,cDiJ->aBiJ',eri_ab[noa:,nob:,noa:,nob:],tab)
    dtbb += 0.5 * einsum('ABCD,CDIJ->ABIJ',eri_bb[nob:,nob:,nob:,nob:],tbb)
    dtaa += 0.5 * einsum('klij,abkl->abij',loooo,taa)
    dtab +=       einsum('kLiJ,aBkL->aBiJ',loOoO,tab)
    dtbb += 0.5 * einsum('KLIJ,ABKL->ABIJ',lOOOO,tbb)

    dlaa += 0.5 * einsum('cdab,ijcd->ijab',lvvvv,laa)
    dlab +=       einsum('cDaB,iJcD->iJaB',lvVvV,lab)
    dlbb += 0.5 * einsum('CDAB,IJCD->IJAB',lVVVV,lbb)
    dlaa += 0.5 * einsum('ijkl,klab->ijab',loooo,laa)
    dlab +=       einsum('iJkL,kLaB->iJaB',loOoO,lab)
    dlbb += 0.5 * einsum('IJKL,KLAB->IJAB',lOOOO,lbb)

    tmp  = einsum('bkjc,acik->abij',eri_aa[noa:,:noa,:noa,noa:],taa)
    tmp += einsum('bKjC,aCiK->abij',eri_ab[noa:,:nob,:noa,nob:],tab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dtaa += tmp.copy()
    tmp  = einsum('BKJC,ACIK->ABIJ',eri_bb[nob:,:nob,:nob,nob:],tbb)
    tmp += einsum('kBcJ,cAkI->ABIJ',eri_ab[:noa,nob:,noa:,:nob],tab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dtbb += tmp.copy()
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
    tmp  = einsum('JCBK,IKAC->IJAB',eri_bb[:nob,nob:,nob:,:nob],lbb)
    tmp += einsum('cJkB,kIcA->IJAB',eri_ab[noa:,:nob,:noa,nob:],lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlbb += tmp.copy()
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
    roVoV  = einsum('kLdC,dBiL->kBiC',eri_ab[:noa,:nob,noa:,nob:],tab)
    rvOvO  = einsum('lKcD,bDlI->bKcI',eri_ab[:noa,:nob,noa:,nob:],tab)
    rOVVO  = einsum('KLCD,BDJL->KBCJ',eri_bb[:nob,:nob,nob:,nob:],tbb)
    rOVVO += einsum('lKdC,dBlJ->KBCJ',eri_ab[:noa,:nob,noa:,nob:],tab)

    tmp  = einsum('kbcj,acik->abij',rovvo,taa)
    tmp += einsum('bKjC,aCiK->abij',rvOoV,tab)
    tmp -= tmp.transpose(0,1,3,2)
    dtaa += tmp.copy()
    tmp  = einsum('KBCJ,ACIK->ABIJ',rOVVO,tbb)
    tmp += einsum('kBcJ,cAkI->ABIJ',roVvO,tab)
    tmp -= tmp.transpose(0,1,3,2)
    dtbb += tmp.copy()
    dtab += einsum('kBcJ,acik->aBiJ',roVvO,taa)
    dtab += einsum('KBCJ,aCiK->aBiJ',rOVVO,tab)
    dtab += einsum('kBiC,aCkJ->aBiJ',roVoV,tab)

    tmp  = einsum('jcbk,ikac->ijab',rovvo,laa)
    tmp += einsum('jCbK,iKaC->ijab',roVvO,lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlaa += tmp.copy()
    tmp  = einsum('JCBK,IKAC->IJAB',rOVVO,lbb)
    tmp += einsum('cJkB,kIcA->IJAB',rvOoV,lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlbb += tmp.copy()
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
    FOO  = 0.5 * einsum('ILCD,CDKL->IK',lbb,tbb)
    FOO +=       einsum('lIdC,dClK->IK',lab,tab)
    FVV  = 0.5 * einsum('KLAD,CDKL->CA',lbb,tbb)
    FVV +=       einsum('lKdA,dClK->CA',lab,tab)

    dlaa -= einsum('ik,kjab->ijab',Foo,eri_aa[:noa,:noa,noa:,noa:])
    dlaa -= einsum('jk,ikab->ijab',Foo,eri_aa[:noa,:noa,noa:,noa:])
    dlaa -= einsum('ca,ijcb->ijab',Fvv,eri_aa[:noa,:noa,noa:,noa:])
    dlaa -= einsum('cb,ijac->ijab',Fvv,eri_aa[:noa,:noa,noa:,noa:])

    dlab -= einsum('ik,kJaB->iJaB',Foo,eri_ab[:noa,:nob,noa:,nob:])
    dlab -= einsum('JK,iKaB->iJaB',FOO,eri_ab[:noa,:nob,noa:,nob:])
    dlab -= einsum('ca,iJcB->iJaB',Fvv,eri_ab[:noa,:nob,noa:,nob:])
    dlab -= einsum('CB,iJaC->iJaB',FVV,eri_ab[:noa,:nob,noa:,nob:])

    dlbb -= einsum('IK,KJAB->IJAB',FOO,eri_bb[:nob,:nob,nob:,nob:])
    dlbb -= einsum('JK,IKAB->IJAB',FOO,eri_bb[:nob,:nob,nob:,nob:])
    dlbb -= einsum('CA,IJCB->IJAB',FVV,eri_bb[:nob,:nob,nob:,nob:])
    dlbb -= einsum('CB,IJAC->IJAB',FVV,eri_bb[:nob,:nob,nob:,nob:])
    return (dtaa, dtab, dtbb), (dlaa, dlab, dlbb)

def compute_gamma(t, l): # normal ordered, asymmetric
    taa, tab, tbb = t
    laa, lab, lbb = l
    dvv  = 0.5 * einsum('ikac,bcik->ba',laa,taa)
    dvv +=       einsum('iKaC,bCiK->ba',lab,tab)
    doo  = 0.5 * einsum('jkac,acik->ji',laa,taa)
    doo +=       einsum('jKaC,aCiK->ji',lab,tab)
    dVV  = 0.5 * einsum('IKAC,BCIK->BA',lbb,tbb)
    dVV +=       einsum('kIcA,cBkI->BA',lab,tab)
    dOO  = 0.5 * einsum('JKAC,ACIK->JI',lbb,tbb)
    dOO +=       einsum('kJcA,cAkI->JI',lab,tab)
    doo *= - 1.0
    dOO *= - 1.0

    dovvo  = einsum('jkbc,acik->jabi',laa,taa)
    dovvo += einsum('jKbC,aCiK->jabi',lab,tab)
    doVvO  = einsum('jkbc,cAkI->jAbI',laa,tab)
    doVvO += einsum('jKbC,ACIK->jAbI',lab,tbb)
    dvOoV  = einsum('kJcB,acik->aJiB',lab,taa)
    dvOoV += einsum('JKBC,aCiK->aJiB',lbb,tab)
    doVoV  = - einsum('jKcB,cAiK->jAiB',lab,tab)
    dvOvO  = - einsum('kJbC,aCkI->aJbI',lab,tab) 
    dOVVO  = einsum('JKBC,ACIK->JABI',lbb,tbb)
    dOVVO += einsum('kJcB,cAkI->JABI',lab,tab)

    dvvvv = 0.5 * einsum('ijab,cdij->cdab',laa,taa)
    dvVvV =       einsum('iJaB,cDiJ->cDaB',lab,tab)
    dVVVV = 0.5 * einsum('IJAB,CDIJ->CDAB',lbb,tbb)
    doooo = 0.5 * einsum('klab,abij->klij',laa,taa)
    doOoO =       einsum('kLaB,aBiJ->kLiJ',lab,tab)
    dOOOO = 0.5 * einsum('KLAB,ABIJ->KLIJ',lbb,tbb)

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

    dVVOO = tbb.copy()
    tmp  = einsum('LADI,BDJL->ABIJ',dOVVO,tbb)
    tmp += einsum('lAdI,dBlJ->ABIJ',doVvO,tab)
    tmp -= tmp.transpose(0,1,3,2)
    dVVOO += tmp.copy()
    dVVOO -= einsum('AC,CBIJ->ABIJ',dVV,tbb)
    dVVOO -= einsum('BC,ACIJ->ABIJ',dVV,tbb)
    dVVOO += einsum('KI,ABKJ->ABIJ',dOO,tbb)
    dVVOO += einsum('KJ,ABIK->ABIJ',dOO,tbb)
    dVVOO += 0.5 * einsum('KLIJ,ABKL->ABIJ',dOOOO,tbb)

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

        dOOOO += einsum('ki,lj->klij',np.eye(nob),dOO)
        dOOOO += einsum('lj,ki->klij',np.eye(nob),dOO)
        dOOOO -= einsum('li,kj->klij',np.eye(nob),dOO)
        dOOOO -= einsum('kj,li->klij',np.eye(nob),dOO)
        dOOOO += einsum('ki,lj->klij',np.eye(nob),np.eye(nob))
        dOOOO -= einsum('li,kj->klij',np.eye(nob),np.eye(nob))

        dovvo -= einsum('ji,ab->jabi',np.eye(noa),dvv)
        doVoV += einsum('ji,AB->jAiB',np.eye(noa),dVV)
        dvOvO += einsum('JI,ab->aJbI',np.eye(nob),dvv)
        dOVVO -= einsum('ji,ab->jabi',np.eye(nob),dVV)

        doo += np.eye(noa)
        dOO += np.eye(nob)

    da = np.zeros((nmoa,nmoa),dtype=complex)
    db = np.zeros((nmob,nmob),dtype=complex)
    da[:noa,:noa] = doo.copy()
    da[noa:,noa:] = dvv.copy()
    db[:nob,:nob] = dOO.copy()
    db[nob:,nob:] = dVV.copy()
    daa = np.zeros((nmoa,nmoa,nmoa,nmoa),dtype=complex)
    daa[:noa,:noa,:noa,:noa] = doooo.copy()
    daa[:noa,:noa,noa:,noa:] = doovv.copy()
    daa[noa:,noa:,:noa,:noa] = dvvoo.copy()
    daa[:noa,noa:,noa:,:noa] = dovvo.copy()
    daa[noa:,:noa,:noa,noa:] =   dovvo.transpose(1,0,3,2).copy() 
    daa[:noa,noa:,:noa,noa:] = - dovvo.transpose(0,1,3,2).copy() 
    daa[noa:,:noa,noa:,:noa] = - dovvo.transpose(1,0,2,3).copy() 
    daa[noa:,noa:,noa:,noa:] = dvvvv.copy()
    dbb = np.zeros((nmob,nmob,nmob,nmob),dtype=complex)
    dbb[:nob,:nob,:nob,:nob] = dOOOO.copy()
    dbb[:nob,:nob,nob:,nob:] = dOOVV.copy()
    dbb[nob:,nob:,:nob,:nob] = dVVOO.copy()
    dbb[:nob,nob:,nob:,:nob] = dOVVO.copy()
    dbb[nob:,:nob,:nob,nob:] =   dOVVO.transpose(1,0,3,2).copy() 
    dbb[:nob,nob:,:nob,nob:] = - dOVVO.transpose(0,1,3,2).copy() 
    dbb[nob:,:nob,nob:,:nob] = - dOVVO.transpose(1,0,2,3).copy() 
    dbb[nob:,nob:,nob:,nob:] = dVVVV.copy()
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
        db = 0.5 * (db + db.T.conj())
        daa = 0.5 * (daa + daa.transpose(2,3,0,1).conj())
        dbb = 0.5 * (dbb + dbb.transpose(2,3,0,1).conj())
        dab = 0.5 * (dab + dab.transpose(2,3,0,1).conj())
    return (da, db), (daa, dab, dbb)

def compute_kappa_intermediates(da, daa, dab, ha, eri_aa, eri_ab, noa):
    nmoa = da.shape[0]
    nva = nmoa - noa

    Cov  = einsum('ba,aj->jb',da[noa:,noa:],ha[noa:,:noa]) 
    Cov -= einsum('ij,bi->jb',da[:noa,:noa],ha[noa:,:noa])
    Cov += 0.5 * einsum('pqjs,bspq->jb',eri_aa[:,:,:noa,:],daa[noa:,:,:,:])
    Cov +=       einsum('pQjS,bSpQ->jb',eri_ab[:,:,:noa,:],dab[noa:,:,:,:])
    Cov -= 0.5 * einsum('bqrs,rsjq->jb',eri_aa[noa:,:,:,:],daa[:,:,:noa,:])
    Cov -=       einsum('bQrS,rSjQ->jb',eri_ab[noa:,:,:,:],dab[:,:,:noa,:])

    Aovvo  = einsum('ba,ij->jbai',np.eye(nva),da[:noa,:noa])
    Aovvo -= einsum('ij,ba->jbai',np.eye(noa),da[noa:,noa:])

    Aovvo = Aovvo.reshape(noa*nva,noa*nva)
    Cov = Cov.reshape(noa*nva)
    kappa = np.dot(np.linalg.inv(Aovvo),Cov)
    kappa = kappa.reshape(nva,noa)
    kappa = np.block([[np.zeros((noa,noa)),-kappa.T.conj()],
                      [kappa, np.zeros((nva,nva))]])
    return kappa

def compute_kappa(d1, d2, eris, no):
    ka = compute_kappa_intermediates(d1[0], d2[0], d2[1], 
         eris.h[0], eris.eri[0], eris.eri[1], no[0])
    kb = compute_kappa_intermediates(d1[1], d2[2], d2[1].transpose(1,0,3,2), 
         eris.h[1], eris.eri[2], eris.eri[1].transpose(1,0,3,2), no[1])
    return ka, kb

def compute_energy(d1, d2, eris):
    ha, hb = eris.h
    eri_aa, eri_ab, eri_bb = eris.eri
    da, db = d1
    daa, dab, dbb = d2

    e  = einsum('pq,qp',ha,da)
    e += einsum('PQ,QP',hb,db)
    e += 0.25 * einsum('pqrs,rspq',eri_aa,daa)
    e += 0.25 * einsum('PQRS,RSPQ',eri_bb,dbb)
    e +=        einsum('pQrS,rSpQ',eri_ab,dab)
    return e.real

def init_amps(eris, mo_coeff, no):
    noa, nob = no
    eris.ao2mo(mo_coeff)
    fa, fb = eris.h[0].copy(), eris.h[1].copy()
    fa += einsum('piqi->pq',eris.eri[0][:,:noa,:,:noa])
    fa += einsum('pIqI->pq',eris.eri[1][:,:nob,:,:nob])
    fb += einsum('PIQI->PQ',eris.eri[2][:,:nob,:,:nob])
    fb += einsum('iPiQ->PQ',eris.eri[1][:noa,:,:noa,:])
    eoa = np.diag(fa[:noa,:noa])
    eva = np.diag(fa[noa:,noa:])
    eob = np.diag(fb[:nob,:nob])
    evb = np.diag(fb[nob:,nob:])
    eia = lib.direct_sum('i-a->ia', eoa, eva)
    eIA = lib.direct_sum('I-A->IA', eob, evb)
    eabij = lib.direct_sum('ia+jb->abij', eia, eia)
    eaBiJ = lib.direct_sum('ia+JB->aBiJ', eia, eIA)
    eABIJ = lib.direct_sum('IA+JB->ABIJ', eIA, eIA)
    taa = eris.eri[0][noa:,noa:,:noa,:noa]/eabij
    tab = eris.eri[1][noa:,nob:,:noa,:nob]/eaBiJ
    tbb = eris.eri[2][nob:,nob:,:nob,:nob]/eABIJ
    laa = taa.transpose(2,3,0,1).copy()
    lab = tab.transpose(2,3,0,1).copy()
    lbb = tbb.transpose(2,3,0,1).copy()
    return (taa, tab, tbb), (laa, lab, lbb)

def kernel_it(mf, maxiter=1000, step=0.03, thresh=1e-8):
    noa, nob = mf.mol.nelec
    eris = ERIs(mf)
    mo0 = mf.mo_coeff.copy()
    Ua = np.eye(mf.mo_coeff[0].shape[0])
    Ub = np.eye(mf.mo_coeff[1].shape[0])
    mo_coeff = np.dot(mo0,Ua), np.dot(mo0,Ub)
    (taa, tab, tbb), (laa, lab, lbb) = init_amps(eris, mo_coeff, mf.mol.nelec)
    d1, d2 = compute_rdms((taa, tab, tbb), (laa, lab, lbb))
    e = compute_energy(d1, d2, eris)

    converged = False
    for i in range(maxiter):
        eris.ao2mo(mo_coeff)
        dt, dl = update_amps((taa, tab, tbb), (laa, lab, lbb), eris)
        taa -= step * dt[0]
        tab -= step * dt[1]
        tbb -= step * dt[2]
        laa -= step * dl[0]
        lab -= step * dl[1]
        lbb -= step * dl[2]
        d1, d2 = compute_rdms((taa, tab, tbb), (laa, lab, lbb))
        e_new = compute_energy(d1, d2, eris)
        de, e = e_new - e, e_new
        ka, kb = compute_kappa(d1, d2, eris, mf.mol.nelec)
        dnormk  = np.linalg.norm(ka) + np.linalg.norm(kb)
        dnormt  = np.linalg.norm(dt[0])
        dnormt += np.linalg.norm(dt[1])
        dnormt += np.linalg.norm(dt[2])
        dnorml  = np.linalg.norm(dl[0])
        dnorml += np.linalg.norm(dl[1])
        dnorml += np.linalg.norm(dl[2])
        print('iter: {}, dk: {}, dt: {}, dl: {}, de: {}, energy: {}'.format(
              i, dnormk, dnormt, dnorml, de, e))
        if dnormk < thresh:
            converged = True
            break
        Ua = np.dot(Ua, scipy.linalg.expm(step*ka)) # U = U_{old,new}
        Ub = np.dot(Ub, scipy.linalg.expm(step*kb)) # U = U_{old,new}
        mo_coeff = np.dot(mo0,Ua), np.dot(mo0,Ub)
    return (taa, tab, tbb), (laa, lab, lbb), (Ua, Ub), e 

class ERIs:
    def __init__(self, mf):
        self.hao = mf.get_hcore().astype(complex)
        self.eri_ao = mf.mol.intor('int2e_sph').astype(complex)

    def ao2mo(self, mo_coeff):
        moa, mob = mo_coeff
        nmoa, nmob = moa.shape[0], mob.shape[0]
    
        ha = einsum('uv,up,vq->pq',self.hao,moa.conj(),moa)
        hb = einsum('uv,up,vq->pq',self.hao,mob.conj(),mob)
        self.h = ha, hb
    
        eri_aa = einsum('uvxy,up,vr->prxy',self.eri_ao,moa.conj(),moa)
        eri_aa = einsum('prxy,xq,ys->prqs',eri_aa,     moa.conj(),moa)
        eri_aa = eri_aa.transpose(0,2,1,3)
        eri_aa = eri_aa - eri_aa.transpose(0,1,3,2)
        eri_bb = einsum('uvxy,up,vr->prxy',self.eri_ao,mob.conj(),mob)
        eri_bb = einsum('prxy,xq,ys->prqs',eri_bb,     mob.conj(),mob)
        eri_bb = eri_bb.transpose(0,2,1,3)
        eri_bb = eri_bb - eri_bb.transpose(0,1,3,2)
        eri_ab = einsum('uvxy,up,vr->prxy',self.eri_ao,moa.conj(),moa)
        eri_ab = einsum('prxy,xq,ys->prqs',eri_ab,     mob.conj(),mob)
        eri_ab = eri_ab.transpose(0,2,1,3)
        self.eri = eri_aa.copy(), eri_ab.copy(), eri_bb.copy()

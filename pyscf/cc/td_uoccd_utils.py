import numpy as np
import scipy
from pyscf import lib, ao2mo
einsum = lib.einsum

def update_amps(t, l, eris, time):
    eris.make_tensors(time)
    taa, tab, tbb = t
    laa, lab, lbb = l
#    eri_aa, eri_ab, eri_bb = eris.eri
#    _, _, noa, nob = tab.shape
    fooa, foob = eris.foo
    fvva, fvvb = eris.fvv

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

    if it: 
        dtaa *= - 1.0
        dtab *= - 1.0
        dtbb *= - 1.0
        dlaa *= - 1.0
        dlab *= - 1.0
        dlbb *= - 1.0
    else: 
        dtaa *= - 1j
        dtab *= - 1j
        dtbb *= - 1j
        dlaa *= 1j
        dlab *= 1j
        dlbb *= 1j
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

def compute_X_intermediates(da, daa, dab, ha, eri_aa, eri_ab, noa, it):
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
    Xvo = np.dot(np.linalg.inv(Aovvo),Cov)
    Xvo = Xvo.reshape(nva,noa)
    Cov = Cov.reshape(noa,nva)
    if it: 
        X = np.block([[np.zeros((noa,noa)),-Xvo.T.conj()],
                      [Xvo, np.zeros((nva,nva))]])
    else:
        X = 1j*np.block([[np.zeros((no,no)),Xvo.T.conj()],
                          [Xvo, np.zeros((nv,nv))]])
        Cov *= 1j
    return X, Cov.T

def compute_X(d1, d2, eris, no, it):
    Xa, Cvoa = compute_X_intermediates(d1[0], d2[0], d2[1], 
         eris.h[0], eris.eri[0], eris.eri[1], no[0], it)
    Xb, Cvob = compute_X_intermediates(d1[1], d2[2], d2[1].transpose(1,0,3,2), 
         eris.h[1], eris.eri[2], eris.eri[1].transpose(1,0,3,2), no[1], it)
    return (Xa, Xb), (Cvoa, Cvob)

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

def update_RK4(t, l, eris, step, it):
    dt1, dl1 = update_amps(t, l, eris, it) 
    t1 = t[0] + dt1[0]*step*0.5, t[1] + dt1[1]*step*0.5, t[2] + dt1[2]*step*0.5 
    l1 = l[0] + dl1[0]*step*0.5, l[1] + dl1[1]*step*0.5, l[2] + dl1[2]*step*0.5 
    dt2, dl2 = update_amps(t1, l1, eris, it) 
    t2 = t[0] + dt2[0]*step*0.5, t[1] + dt2[1]*step*0.5, t[2] + dt2[2]*step*0.5 
    l2 = l[0] + dl2[0]*step*0.5, l[1] + dl2[1]*step*0.5, l[2] + dl2[2]*step*0.5 
    dt3, dl3 = update_amps(t2, l2, eris, it) 
    t3 = t[0] + dt3[0]*step, t[1] + dt3[1]*step, t[2] + dt3[2]*step 
    l3 = l[0] + dl3[0]*step, l[1] + dl3[1]*step, l[2] + dl3[2]*step 
    dt4, dl4 = update_amps(t3, l3, eris, it) 

    dtaa = (dt1[0] + 2.0*dt2[0] + 2.0*dt3[0] + dt4[0])/6.0
    dtab = (dt1[1] + 2.0*dt2[1] + 2.0*dt3[1] + dt4[1])/6.0
    dtbb = (dt1[2] + 2.0*dt2[2] + 2.0*dt3[2] + dt4[2])/6.0
    dlaa = (dl1[0] + 2.0*dl2[0] + 2.0*dl3[0] + dl4[0])/6.0
    dlab = (dl1[1] + 2.0*dl2[1] + 2.0*dl3[1] + dl4[1])/6.0
    dlbb = (dl1[2] + 2.0*dl2[2] + 2.0*dl3[2] + dl4[2])/6.0
    return (dtaa, dtab, dtbb), (dlaa, dlab, dlbb)


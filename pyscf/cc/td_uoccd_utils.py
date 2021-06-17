import numpy as np
import scipy
from pyscf import lib, ao2mo, cc
from pyscf.cc import td_roccd_utils
from lattice.fci import FCISimple
einsum = lib.einsum

def update_amps(t, l, eris, time):
    eris.make_tensors(time)
    taa, tab, tbb = t
    laa, lab, lbb = l

    Foo  = eris.foo.copy()
    Foo += 0.5 * einsum('klcd,cdjl->kj',eris.oovv,taa)
    Foo +=       einsum('kLcD,cDjL->kj',eris.oOvV,tab)
    Fvv  = eris.fvv.copy()
    Fvv -= 0.5 * einsum('klcd,bdkl->bc',eris.oovv,taa)
    Fvv -=       einsum('kLcD,bDkL->bc',eris.oOvV,tab)

    FOO  = eris.fOO.copy()
    FOO += 0.5 * einsum('KLCD,CDJL->KJ',eris.OOVV,tbb)
    FOO +=       einsum('lKdC,dClJ->KJ',eris.oOvV,tab)
    FVV  = eris.fVV.copy()
    FVV -= 0.5 * einsum('KLCD,BDKL->BC',eris.OOVV,tbb)
    FVV -=       einsum('lKdC,dBlK->BC',eris.oOvV,tab)

    dtaa  = eris.oovv.transpose(2,3,0,1).conj().copy()
    dtaa += einsum('bc,acij->abij',Fvv,taa)
    dtaa += einsum('ac,cbij->abij',Fvv,taa)
    dtaa -= einsum('kj,abik->abij',Foo,taa)
    dtaa -= einsum('ki,abkj->abij',Foo,taa)

    dtab  = eris.oOvV.transpose(2,3,0,1).conj().copy()
    dtab += einsum('BC,aCiJ->aBiJ',FVV,tab)
    dtab += einsum('ac,cBiJ->aBiJ',Fvv,tab)
    dtab -= einsum('KJ,aBiK->aBiJ',FOO,tab)
    dtab -= einsum('ki,aBkJ->aBiJ',Foo,tab)

    dtbb  = eris.OOVV.transpose(2,3,0,1).conj().copy()
    dtbb += einsum('BC,ACIJ->ABIJ',FVV,tbb)
    dtbb += einsum('AC,CBIJ->ABIJ',FVV,tbb)
    dtbb -= einsum('KJ,ABIK->ABIJ',FOO,tbb)
    dtbb -= einsum('KI,ABKJ->ABIJ',FOO,tbb)

    dlaa  = eris.oovv.copy()
    dlaa += einsum('cb,ijac->ijab',Fvv,laa)
    dlaa += einsum('ca,ijcb->ijab',Fvv,laa)
    dlaa -= einsum('jk,ikab->ijab',Foo,laa)
    dlaa -= einsum('ik,kjab->ijab',Foo,laa)

    dlab  = eris.oOvV.copy()
    dlab += einsum('CB,iJaC->iJaB',FVV,lab)
    dlab += einsum('ca,iJcB->iJaB',Fvv,lab)
    dlab -= einsum('JK,iKaB->iJaB',FOO,lab)
    dlab -= einsum('ik,kJaB->iJaB',Foo,lab)

    dlbb  = eris.OOVV.copy()
    dlbb += einsum('CB,IJAC->IJAB',FVV,lbb)
    dlbb += einsum('CA,IJCB->IJAB',FVV,lbb)
    dlbb -= einsum('JK,IKAB->IJAB',FOO,lbb)
    dlbb -= einsum('IK,KJAB->IJAB',FOO,lbb)

    loooo  = eris.oooo.copy()
    loooo += 0.5 * einsum('klcd,cdij->klij',eris.oovv,taa)
    loOoO  = eris.oOoO.copy()
    loOoO +=       einsum('kLcD,cDiJ->kLiJ',eris.oOvV,tab)
    lOOOO  = eris.OOOO.copy()
    lOOOO += 0.5 * einsum('KLCD,CDIJ->KLIJ',eris.OOVV,tbb)
    lvvvv  = eris.vvvv.copy()
    lvvvv += 0.5 * einsum('klab,cdkl->cdab',eris.oovv,taa)
    lvVvV  = eris.vVvV.copy()
    lvVvV +=       einsum('kLaB,cDkL->cDaB',eris.oOvV,tab)
    lVVVV  = eris.VVVV.copy()
    lVVVV += 0.5 * einsum('KLAB,CDKL->CDAB',eris.OOVV,tbb)

    dtaa += 0.5 * einsum('abcd,cdij->abij',eris.vvvv,taa)
    dtab +=       einsum('aBcD,cDiJ->aBiJ',eris.vVvV,tab)
    dtbb += 0.5 * einsum('ABCD,CDIJ->ABIJ',eris.VVVV,tbb)
    dtaa += 0.5 * einsum('klij,abkl->abij',loooo,taa)
    dtab +=       einsum('kLiJ,aBkL->aBiJ',loOoO,tab)
    dtbb += 0.5 * einsum('KLIJ,ABKL->ABIJ',lOOOO,tbb)

    dlaa += 0.5 * einsum('cdab,ijcd->ijab',lvvvv,laa)
    dlab +=       einsum('cDaB,iJcD->iJaB',lvVvV,lab)
    dlbb += 0.5 * einsum('CDAB,IJCD->IJAB',lVVVV,lbb)
    dlaa += 0.5 * einsum('ijkl,klab->ijab',loooo,laa)
    dlab +=       einsum('iJkL,kLaB->iJaB',loOoO,lab)
    dlbb += 0.5 * einsum('IJKL,KLAB->IJAB',lOOOO,lbb)

    rovvo  = einsum('klcd,bdjl->kbcj',eris.oovv,taa)
    rovvo += einsum('kLcD,bDjL->kbcj',eris.oOvV,tab)
    rvOoV  = einsum('lKdC,bdjl->bKjC',eris.oOvV,taa)
    rvOoV += einsum('KLCD,bDjL->bKjC',eris.OOVV,tab)
    roVvO  = einsum('kLcD,BDJL->kBcJ',eris.oOvV,tbb)
    roVvO += einsum('klcd,dBlJ->kBcJ',eris.oovv,tab)
    roVoV  = einsum('kLdC,dBiL->kBiC',eris.oOvV,tab)
    rvOvO  = einsum('lKcD,bDlI->bKcI',eris.oOvV,tab)
    rOVVO  = einsum('KLCD,BDJL->KBCJ',eris.OOVV,tbb)
    rOVVO += einsum('lKdC,dBlJ->KBCJ',eris.oOvV,tab)
    vOoV = eris.oVvO.transpose(2,3,0,1).conj().copy()

    tmp  = einsum('kbcj,acik->abij',eris.ovvo+0.5*rovvo,taa)
    tmp += einsum('bKjC,aCiK->abij',     vOoV+0.5*rvOoV,tab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dtaa += tmp.copy()
    tmp  = einsum('KBCJ,ACIK->ABIJ',eris.OVVO+0.5*rOVVO,tbb)
    tmp += einsum('kBcJ,cAkI->ABIJ',eris.oVvO+0.5*roVvO,tab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dtbb += tmp.copy()
    dtab += einsum('kBcJ,acik->aBiJ',eris.oVvO+roVvO,taa)
    dtab += einsum('KBCJ,aCiK->aBiJ',eris.OVVO+rOVVO,tab)
    dtab -= einsum('kBiC,aCkJ->aBiJ',eris.oVoV-roVoV,tab)
    dtab -= einsum('aKcJ,cBiK->aBiJ',eris.vOvO,tab)
    dtab += einsum('kaci,cBkJ->aBiJ',eris.ovvo,tab)
    dtab += einsum('aKiC,BCJK->aBiJ',     vOoV,tbb)

    tmp  = einsum('jcbk,ikac->ijab',eris.ovvo+rovvo,laa)
    tmp += einsum('jCbK,iKaC->ijab',eris.oVvO+roVvO,lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlaa += tmp.copy()
    tmp  = einsum('JCBK,IKAC->IJAB',eris.OVVO+rOVVO,lbb)
    tmp += einsum('cJkB,kIcA->IJAB',     vOoV+rvOoV,lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlbb += tmp.copy()
    dlab += einsum('cJkB,ikac->iJaB',     vOoV+rvOoV,laa)
    dlab += einsum('JCBK,iKaC->iJaB',eris.OVVO+rOVVO,lab)
    dlab -= einsum('cJaK,iKcB->iJaB',eris.vOvO-rvOvO,lab)
    dlab -= einsum('iCkB,kJaC->iJaB',eris.oVoV-roVoV,lab)
    dlab += einsum('icak,kJcB->iJaB',eris.ovvo+rovvo,lab)
    dlab += einsum('iCaK,JKBC->iJaB',eris.oVvO+roVvO,lbb)

    Foo  = 0.5 * einsum('ilcd,cdkl->ik',laa,taa)
    Foo +=       einsum('iLcD,cDkL->ik',lab,tab)
    Fvv  = 0.5 * einsum('klad,cdkl->ca',laa,taa)
    Fvv +=       einsum('kLaD,cDkL->ca',lab,tab)
    FOO  = 0.5 * einsum('ILCD,CDKL->IK',lbb,tbb)
    FOO +=       einsum('lIdC,dClK->IK',lab,tab)
    FVV  = 0.5 * einsum('KLAD,CDKL->CA',lbb,tbb)
    FVV +=       einsum('lKdA,dClK->CA',lab,tab)

    dlaa -= einsum('ik,kjab->ijab',Foo,eris.oovv)
    dlaa -= einsum('jk,ikab->ijab',Foo,eris.oovv)
    dlaa -= einsum('ca,ijcb->ijab',Fvv,eris.oovv)
    dlaa -= einsum('cb,ijac->ijab',Fvv,eris.oovv)

    dlab -= einsum('ik,kJaB->iJaB',Foo,eris.oOvV)
    dlab -= einsum('JK,iKaB->iJaB',FOO,eris.oOvV)
    dlab -= einsum('ca,iJcB->iJaB',Fvv,eris.oOvV)
    dlab -= einsum('CB,iJaC->iJaB',FVV,eris.oOvV)

    dlbb -= einsum('IK,KJAB->IJAB',FOO,eris.OOVV)
    dlbb -= einsum('JK,IKAB->IJAB',FOO,eris.OOVV)
    dlbb -= einsum('CA,IJCB->IJAB',FVV,eris.OOVV)
    dlbb -= einsum('CB,IJAC->IJAB',FVV,eris.OOVV)

    loooo = loOoO = lOOOO = lvvvv = lvVvV = lVVVV = None
    rovov = rvOoV = roVvO = roVoV = rvOvO = rOVVO = None
    vOoV = tmp = Foo = Fvv = None
    return (-1j*dtaa, -1j*dtab, -1j*dtbb), (1j*dlaa, 1j*dlab, 1j*dlbb)

def update_amps_quick(t, l, eris, time): 
    # for hubbard and SIAM, where 2e part of H has only ab-component
    eris.make_tensors(time)
    taa, tab, tbb = t
    laa, lab, lbb = l

    Foo  = eris.foo.copy()
#    Foo += 0.5 * einsum('klcd,cdjl->kj',eris.oovv,taa)
    Foo +=       einsum('kLcD,cDjL->kj',eris.oOvV,tab)
    Fvv  = eris.fvv.copy()
#    Fvv -= 0.5 * einsum('klcd,bdkl->bc',eris.oovv,taa)
    Fvv -=       einsum('kLcD,bDkL->bc',eris.oOvV,tab)

    FOO  = eris.fOO.copy()
#    FOO += 0.5 * einsum('KLCD,CDJL->KJ',eris.OOVV,tbb)
    FOO +=       einsum('lKdC,dClJ->KJ',eris.oOvV,tab)
    FVV  = eris.fVV.copy()
#    FVV -= 0.5 * einsum('KLCD,BDKL->BC',eris.OOVV,tbb)
    FVV -=       einsum('lKdC,dBlK->BC',eris.oOvV,tab)

#    dtaa  = eris.oovv.transpose(2,3,0,1).conj().copy()
    dtaa  = einsum('bc,acij->abij',Fvv,taa)
    dtaa += einsum('ac,cbij->abij',Fvv,taa)
    dtaa -= einsum('kj,abik->abij',Foo,taa)
    dtaa -= einsum('ki,abkj->abij',Foo,taa)

    dtab  = eris.oOvV.transpose(2,3,0,1).conj().copy()
    dtab += einsum('BC,aCiJ->aBiJ',FVV,tab)
    dtab += einsum('ac,cBiJ->aBiJ',Fvv,tab)
    dtab -= einsum('KJ,aBiK->aBiJ',FOO,tab)
    dtab -= einsum('ki,aBkJ->aBiJ',Foo,tab)

#    dtbb  = eris.OOVV.transpose(2,3,0,1).conj().copy()
    dtbb  = einsum('BC,ACIJ->ABIJ',FVV,tbb)
    dtbb += einsum('AC,CBIJ->ABIJ',FVV,tbb)
    dtbb -= einsum('KJ,ABIK->ABIJ',FOO,tbb)
    dtbb -= einsum('KI,ABKJ->ABIJ',FOO,tbb)

#    dlaa  = eris.oovv.copy()
    dlaa  = einsum('cb,ijac->ijab',Fvv,laa)
    dlaa += einsum('ca,ijcb->ijab',Fvv,laa)
    dlaa -= einsum('jk,ikab->ijab',Foo,laa)
    dlaa -= einsum('ik,kjab->ijab',Foo,laa)

    dlab  = eris.oOvV.copy()
    dlab += einsum('CB,iJaC->iJaB',FVV,lab)
    dlab += einsum('ca,iJcB->iJaB',Fvv,lab)
    dlab -= einsum('JK,iKaB->iJaB',FOO,lab)
    dlab -= einsum('ik,kJaB->iJaB',Foo,lab)

#    dlbb  = eris.OOVV.copy()
    dlbb  = einsum('CB,IJAC->IJAB',FVV,lbb)
    dlbb += einsum('CA,IJCB->IJAB',FVV,lbb)
    dlbb -= einsum('JK,IKAB->IJAB',FOO,lbb)
    dlbb -= einsum('IK,KJAB->IJAB',FOO,lbb)

#    loooo  = eris.oooo.copy()
#    loooo += 0.5 * einsum('klcd,cdij->klij',eris.oovv,taa)
    loOoO  = eris.oOoO.copy()
    loOoO +=       einsum('kLcD,cDiJ->kLiJ',eris.oOvV,tab)
#    lOOOO  = eris.OOOO.copy()
#    lOOOO += 0.5 * einsum('KLCD,CDIJ->KLIJ',eris.OOVV,tbb)
#    lvvvv  = eris.vvvv.copy()
#    lvvvv += 0.5 * einsum('klab,cdkl->cdab',eris.oovv,taa)
    lvVvV  = eris.vVvV.copy()
    lvVvV +=       einsum('kLaB,cDkL->cDaB',eris.oOvV,tab)
#    lVVVV  = eris.VVVV.copy()
#    lVVVV += 0.5 * einsum('KLAB,CDKL->CDAB',eris.OOVV,tbb)

#    dtaa += 0.5 * einsum('abcd,cdij->abij',eris.vvvv,taa)
    dtab +=       einsum('aBcD,cDiJ->aBiJ',eris.vVvV,tab)
#    dtbb += 0.5 * einsum('ABCD,CDIJ->ABIJ',eris.VVVV,tbb)
#    dtaa += 0.5 * einsum('klij,abkl->abij',loooo,taa)
    dtab +=       einsum('kLiJ,aBkL->aBiJ',loOoO,tab)
#    dtbb += 0.5 * einsum('KLIJ,ABKL->ABIJ',lOOOO,tbb)

#    dlaa += 0.5 * einsum('cdab,ijcd->ijab',lvvvv,laa)
    dlab +=       einsum('cDaB,iJcD->iJaB',lvVvV,lab)
#    dlbb += 0.5 * einsum('CDAB,IJCD->IJAB',lVVVV,lbb)
#    dlaa += 0.5 * einsum('ijkl,klab->ijab',loooo,laa)
    dlab +=       einsum('iJkL,kLaB->iJaB',loOoO,lab)
#    dlbb += 0.5 * einsum('IJKL,KLAB->IJAB',lOOOO,lbb)

#    rovvo  = einsum('klcd,bdjl->kbcj',eris.oovv,taa)
    rovvo  = einsum('kLcD,bDjL->kbcj',eris.oOvV,tab)
    rvOoV  = einsum('lKdC,bdjl->bKjC',eris.oOvV,taa)
#    rvOoV += einsum('KLCD,bDjL->bKjC',eris.OOVV,tab)
    roVvO  = einsum('kLcD,BDJL->kBcJ',eris.oOvV,tbb)
#    roVvO += einsum('klcd,dBlJ->kBcJ',eris.oovv,tab)
    roVoV  = einsum('kLdC,dBiL->kBiC',eris.oOvV,tab)
    rvOvO  = einsum('lKcD,bDlI->bKcI',eris.oOvV,tab)
#    rOVVO  = einsum('KLCD,BDJL->KBCJ',eris.OOVV,tbb)
    rOVVO  = einsum('lKdC,dBlJ->KBCJ',eris.oOvV,tab)
    vOoV = eris.oVvO.transpose(2,3,0,1).conj().copy()

    tmp  = einsum('kbcj,acik->abij',          0.5*rovvo,taa)
    tmp += einsum('bKjC,aCiK->abij',     vOoV+0.5*rvOoV,tab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dtaa += tmp.copy()
    tmp  = einsum('KBCJ,ACIK->ABIJ',          0.5*rOVVO,tbb)
    tmp += einsum('kBcJ,cAkI->ABIJ',eris.oVvO+0.5*roVvO,tab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dtbb += tmp.copy()
    dtab += einsum('kBcJ,acik->aBiJ',eris.oVvO+roVvO,taa)
    dtab += einsum('KBCJ,aCiK->aBiJ',          rOVVO,tab)
    dtab -= einsum('kBiC,aCkJ->aBiJ',eris.oVoV-roVoV,tab)
    dtab -= einsum('aKcJ,cBiK->aBiJ',eris.vOvO,tab)
#    dtab += einsum('kaci,cBkJ->aBiJ',eris.ovvo,tab)
    dtab += einsum('aKiC,BCJK->aBiJ',     vOoV,tbb)

    tmp  = einsum('jcbk,ikac->ijab',          rovvo,laa)
    tmp += einsum('jCbK,iKaC->ijab',eris.oVvO+roVvO,lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlaa += tmp.copy()
    tmp  = einsum('JCBK,IKAC->IJAB',          rOVVO,lbb)
    tmp += einsum('cJkB,kIcA->IJAB',     vOoV+rvOoV,lab)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dlbb += tmp.copy()
    dlab += einsum('cJkB,ikac->iJaB',     vOoV+rvOoV,laa)
    dlab += einsum('JCBK,iKaC->iJaB',          rOVVO,lab)
    dlab -= einsum('cJaK,iKcB->iJaB',eris.vOvO-rvOvO,lab)
    dlab -= einsum('iCkB,kJaC->iJaB',eris.oVoV-roVoV,lab)
    dlab += einsum('icak,kJcB->iJaB',          rovvo,lab)
    dlab += einsum('iCaK,JKBC->iJaB',eris.oVvO+roVvO,lbb)

    Foo  = 0.5 * einsum('ilcd,cdkl->ik',laa,taa)
    Foo +=       einsum('iLcD,cDkL->ik',lab,tab)
    Fvv  = 0.5 * einsum('klad,cdkl->ca',laa,taa)
    Fvv +=       einsum('kLaD,cDkL->ca',lab,tab)
    FOO  = 0.5 * einsum('ILCD,CDKL->IK',lbb,tbb)
    FOO +=       einsum('lIdC,dClK->IK',lab,tab)
    FVV  = 0.5 * einsum('KLAD,CDKL->CA',lbb,tbb)
    FVV +=       einsum('lKdA,dClK->CA',lab,tab)

#    dlaa -= einsum('ik,kjab->ijab',Foo,eris.oovv)
#    dlaa -= einsum('jk,ikab->ijab',Foo,eris.oovv)
#    dlaa -= einsum('ca,ijcb->ijab',Fvv,eris.oovv)
#    dlaa -= einsum('cb,ijac->ijab',Fvv,eris.oovv)

    dlab -= einsum('ik,kJaB->iJaB',Foo,eris.oOvV)
    dlab -= einsum('JK,iKaB->iJaB',FOO,eris.oOvV)
    dlab -= einsum('ca,iJcB->iJaB',Fvv,eris.oOvV)
    dlab -= einsum('CB,iJaC->iJaB',FVV,eris.oOvV)

#    dlbb -= einsum('IK,KJAB->IJAB',FOO,eris.OOVV)
#    dlbb -= einsum('JK,IKAB->IJAB',FOO,eris.OOVV)
#    dlbb -= einsum('CA,IJCB->IJAB',FVV,eris.OOVV)
#    dlbb -= einsum('CB,IJAC->IJAB',FVV,eris.OOVV)

    loooo = loOoO = lOOOO = lvvvv = lvVvV = lVVVV = None
    rovov = rvOoV = roVvO = roVoV = rvOvO = rOVVO = None
    vOoV = tmp = Foo = Fvv = None
    return (-1j*dtaa, -1j*dtab, -1j*dtbb), (1j*dlaa, 1j*dlab, 1j*dlbb)

def compute_rho1(t, l):
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
    return (doo, dOO), (dvv, dVV)

def compute_rho12(t, l): # normal ordered, asymmetric
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
    doovv = laa.copy(), lab.copy(), lbb.copy()
    dvvoo = dvvoo, dvVoO, dVVOO
    dovvo = dovvo, doVvO, dvOoV, doVoV, dvOvO, dOVVO 
    return (doo, dvv), (doooo, doovv, dvvoo, dovvo, dvvvv)

def compute_rdm1(t, l):
    (doo, dOO), (dvv, dVV) = compute_rho1(t, l)
    noa, nob = doo.shape[0], dOO.shape[0]
    doo += np.eye(noa)
    dOO += np.eye(nob)
    
    doo += doo.T.conj()
    dOO += dOO.T.conj()
    dvv += dvv.T.conj()
    dVV += dVV.T.conj()
    doo *= 0.5
    dOO *= 0.5
    dvv *= 0.5
    dVV *= 0.5
    return (doo, dOO), (dvv, dVV)

def compute_rdm12(t, l):
    d1, d2 = compute_rho12(t, l)
    doo, dvv = d1 
    doooo, doovv, dvvoo, dovvo, dvvvv = d2
    doo, dOO = doo
    dvv, dVV = dvv
    doooo, doOoO, dOOOO = doooo
    dvvvv, dvVvV, dVVVV = dvvvv
    doovv, doOvV, dOOVV = doovv
    dvvoo, dvVoO, dVVOO = dvvoo
    dovvo, doVvO, dvOoV, doVoV, dvOvO, dOVVO = dovvo 

    noa, nob, nva, nvb = doOvV.shape
    Ioo, IOO = np.eye(noa), np.eye(nob)
    doooo += einsum('ki,lj->klij',Ioo,doo)
    doooo += einsum('lj,ki->klij',Ioo,doo)
    doooo -= einsum('li,kj->klij',Ioo,doo)
    doooo -= einsum('kj,li->klij',Ioo,doo)
    doooo += einsum('ki,lj->klij',Ioo,Ioo)
    doooo -= einsum('li,kj->klij',Ioo,Ioo)

    doOoO += einsum('ki,LJ->kLiJ',Ioo,dOO)
    doOoO += einsum('LJ,ki->kLiJ',IOO,doo)
    doOoO += einsum('ki,LJ->kLiJ',Ioo,IOO)

    dOOOO += einsum('ki,lj->klij',IOO,dOO)
    dOOOO += einsum('lj,ki->klij',IOO,dOO)
    dOOOO -= einsum('li,kj->klij',IOO,dOO)
    dOOOO -= einsum('kj,li->klij',IOO,dOO)
    dOOOO += einsum('ki,lj->klij',IOO,IOO)
    dOOOO -= einsum('li,kj->klij',IOO,IOO)

    dovvo -= einsum('ji,ab->jabi',Ioo,dvv)
    doVoV += einsum('ji,AB->jAiB',Ioo,dVV)
    dvOvO += einsum('JI,ab->aJbI',IOO,dvv)
    dOVVO -= einsum('JI,AB->JABI',IOO,dVV)

    doo += Ioo
    dOO += IOO

    doo += doo.T.conj()
    dOO += dOO.T.conj()
    dvv += dvv.T.conj()
    dVV += dVV.T.conj()
    doooo += doooo.transpose(2,3,0,1).conj()
    doOoO += doOoO.transpose(2,3,0,1).conj()
    dOOOO += dOOOO.transpose(2,3,0,1).conj()
    doovv += dvvoo.transpose(2,3,0,1).conj()
    doOvV += dvVoO.transpose(2,3,0,1).conj()
    dOOVV += dVVOO.transpose(2,3,0,1).conj()
    dovvo += dovvo.transpose(3,2,1,0).conj()
    doVvO += dvOoV.transpose(2,3,0,1).conj()
    dOVVO += dOVVO.transpose(3,2,1,0).conj()
    doVoV += doVoV.transpose(2,3,0,1).conj()
    dvOvO += dvOvO.transpose(2,3,0,1).conj()
    dvvvv += dvvvv.transpose(2,3,0,1).conj()
    dvVvV += dvVvV.transpose(2,3,0,1).conj()
    dVVVV += dVVVV.transpose(2,3,0,1).conj()
    doo *= 0.5
    dOO *= 0.5
    dvv *= 0.5
    dVV *= 0.5
    doooo *= 0.5
    doOoO *= 0.5
    dOOOO *= 0.5
    doovv *= 0.5
    doOvV *= 0.5
    dOOVV *= 0.5
    dovvo *= 0.5
    doVvO *= 0.5
    dOVVO *= 0.5
    doVoV *= 0.5
    dvOvO *= 0.5
    dvvvv *= 0.5
    dvVvV *= 0.5
    dVVVV *= 0.5
    
    doo = doo, dOO
    dvv = dvv, dVV
    doooo = doooo, doOoO, dOOOO
    dvvvv = dvvvv, dvVvV, dVVVV
    doovv = doovv, doOvV, dOOVV 
    dovvo = dovvo, doVvO, doVoV, dvOvO, dOVVO 
    return (doo, dvv), (doooo, doovv, dovvo, dvvvv)

def compute_Aovvo(doo, dvv):
    no, nv = doo.shape[0], dvv.shape[0]
    Aovvo  = einsum('ab,ji->iabj',np.eye(nv),doo)
    Aovvo -= einsum('ij,ab->iabj',np.eye(no),dvv)
    return Aovvo

def compute_comma(d1, d2, eris, time=None, full=True):
    eris.make_tensors(time)
    doo, dvv = d1
    doooo, doovv, dovvo, dvvvv = d2
    doo, dOO = doo
    dvv, dVV = dvv
    doooo, doOoO, dOOOO = doooo
    dvvvv, dvVvV, dVVVV = dvvvv
    doovv, doOvV, dOOVV = doovv
    dovvo, doVvO, doVoV, dvOvO, dOVVO = dovvo

    fvo  = einsum('ab,ib->ai',dvv,eris.hov.conj())
    fvo += 0.5 * einsum('abcd,ibcd->ai',dvvvv,eris.ovvv.conj())
    fvo += 0.5 * einsum('klab,klib->ai',doovv.conj(),eris.ooov)
    fvo +=       einsum('jabk,ijkb->ai',dovvo,eris.ooov.conj())
    fvo += einsum('aBcD,iBcD->ai',dvVvV,eris.oVvV.conj())
    fvo += einsum('kLaB,kLiB->ai',doOvV.conj(),eris.oOoV)
    fvo += einsum('kBaJ,iJkB->ai',doVvO.conj(),eris.oOoV.conj())
    fvo += einsum('aJbK,iJbK->ai',dvOvO,eris.oOvO.conj())

    fov  = einsum('ij,ja->ia',doo,eris.hov)
    fov += 0.5 * einsum('ijkl,lkja->ia',doooo,eris.ooov)
    fov += 0.5 * einsum('jidc,jadc->ia',doovv,eris.ovvv.conj())
    fov +=       einsum('ibcj,jcba->ia',dovvo,eris.ovvv)
    fov += einsum('iJkL,kLaJ->ia',doOoO,eris.oOvO)
    fov += einsum('iJcD,aJcD->ia',doOvV,eris.vOvV.conj())
    fov += einsum('iBcJ,cJaB->ia',doVvO,eris.vOvV)
    fov += einsum('iBjC,jCaB->ia',doVoV,eris.oVvV)

    foo = fvv = None
    if full:
        foo  = einsum('ik,kj->ij',doo,eris.hoo)
        foo += 0.5 * einsum('imkl,kljm->ij',doooo,eris.oooo)
        foo += 0.5 * einsum('imab,jmab->ij',doovv,eris.oovv.conj())
        foo +=       einsum('iabk,kbaj->ij',dovvo,eris.ovvo)
        foo += einsum('iMkL,kLjM->ij',doOoO,eris.oOoO)
        foo += einsum('iMaB,jMaB->ij',doOvV,eris.oOvV.conj())
        foo += einsum('iAbK,jAbK->ij',doVvO,eris.oVvO.conj())
        foo += einsum('iAkB,kBjA->ij',doVoV,eris.oVoV)

        fvv  = einsum('ac,cb->ab',dvv,eris.hvv)
        fvv += 0.5 * einsum('aecd,cdbe->ab',dvvvv,eris.vvvv)
        fvv += 0.5 * einsum('ijae,ijbe->ab',doovv.conj(),eris.oovv)
        fvv +=       einsum('iacj,jcbi->ab',dovvo,eris.ovvo)
        fvv += einsum('aEcD,cDbE->ab',dvVvV,eris.vVvV)
        fvv += einsum('iJaE,iJbE->ab',doOvV.conj(),eris.oOvV)
        fvv += einsum('jCaI,jCbI->ab',doVvO.conj(),eris.oVvO)
        fvv += einsum('aIcJ,cJbI->ab',dvOvO,eris.vOvO)
    return foo, fov, fvo, fvv

def compute_commb(d1, d2, eris, time=None, full=True):
    eris.make_tensors(time)
    doo, dvv = d1
    doooo, doovv, dovvo, dvvvv = d2
    doo, dOO = doo
    dvv, dVV = dvv
    doooo, doOoO, dOOOO = doooo
    dvvvv, dvVvV, dVVVV = dvvvv
    doovv, doOvV, dOOVV = doovv
    dovvo, doVvO, doVoV, dvOvO, dOVVO = dovvo

    fvo  = einsum('ab,ib->ai',dVV,eris.hOV.conj())
    fvo += 0.5 * einsum('abcd,ibcd->ai',dVVVV,eris.OVVV.conj())
    fvo += 0.5 * einsum('klab,klib->ai',dOOVV.conj(),eris.OOOV)
    fvo +=       einsum('jabk,ijkb->ai',dOVVO,eris.OOOV.conj())
    fvo += einsum('BaDc,BiDc->ai',dvVvV,eris.vOvV.conj())
    fvo += einsum('LkBa,LkBi->ai',doOvV.conj(),eris.oOvO)
    fvo += einsum('JaBk,JiBk->ai',doVvO,eris.oOvO.conj())
    fvo += einsum('JaKb,JiKb->ai',doVoV,eris.oOoV.conj())

    fov  = einsum('ij,ja->ia',dOO,eris.hOV)
    fov += 0.5 * einsum('ijkl,lkja->ia',dOOOO,eris.OOOV)
    fov += 0.5 * einsum('jidc,jadc->ia',dOOVV,eris.OVVV.conj())
    fov +=       einsum('ibcj,jcba->ia',dOVVO,eris.OVVV)
    fov += einsum('JiLk,LkJa->ia',doOoO,eris.oOoV)
    fov += einsum('JiDc,JaDc->ia',doOvV,eris.oVvV.conj())
    fov += einsum('JcBi,JcBa->ia',doVvO.conj(),eris.oVvV)
    fov += einsum('BiCj,CjBa->ia',dvOvO,eris.vOvV)

    foo = fvv = None
    if full:
        foo  = einsum('ik,kj->ij',dOO,eris.hOO)
        foo += 0.5 * einsum('imkl,kljm->ij',dOOOO,eris.OOOO)
        foo += 0.5 * einsum('imab,jmab->ij',dOOVV,eris.OOVV.conj())
        foo +=       einsum('iabk,kbaj->ij',dOVVO,eris.OVVO)
        foo += einsum('MiLk,LkMj->ij',doOoO,eris.oOoO)
        foo += einsum('MiBa,MjBa->ij',doOvV,eris.oOvV.conj())
        foo += einsum('KbAi,KbAj->ij',doVvO.conj(),eris.oVvO)
        foo += einsum('AiBk,BkAj->ij',dvOvO,eris.vOvO)

        fvv  = einsum('ac,cb->ab',dVV,eris.hVV)
        fvv += 0.5 * einsum('aecd,cdbe->ab',dVVVV,eris.VVVV)
        fvv += 0.5 * einsum('ijae,ijbe->ab',dOOVV.conj(),eris.OOVV)
        fvv +=       einsum('iacj,jcbi->ab',dOVVO,eris.OVVO)
        fvv += einsum('EaDc,DcEb->ab',dvVvV,eris.vVvV)
        fvv += einsum('JiEa,JiEb->ab',doOvV.conj(),eris.oOvV)
        fvv += einsum('IaCj,IbCj->ab',doVvO,eris.oVvO.conj())
        fvv += einsum('IaJc,JcIb->ab',doVoV,eris.oVoV)
    return foo, fov, fvo, fvv

def compute_comma_quick(d1, d2, eris, time=None, full=True):
    # for hubbard and SIAM
    eris.make_tensors(time)
    doo, dvv = d1
    doooo, doovv, dovvo, dvvvv = d2
    doo, dOO = doo
    dvv, dVV = dvv
    doooo, doOoO, dOOOO = doooo
    dvvvv, dvVvV, dVVVV = dvvvv
    doovv, doOvV, dOOVV = doovv
    dovvo, doVvO, doVoV, dvOvO, dOVVO = dovvo

    fvo  = einsum('ab,ib->ai',dvv,eris.hov.conj())
#    fvo += 0.5 * einsum('abcd,ibcd->ai',dvvvv,eris.ovvv.conj())
#    fvo += 0.5 * einsum('klab,klib->ai',doovv.conj(),eris.ooov)
#    fvo +=       einsum('jabk,ijkb->ai',dovvo,eris.ooov.conj())
    fvo += einsum('aBcD,iBcD->ai',dvVvV,eris.oVvV.conj())
    fvo += einsum('kLaB,kLiB->ai',doOvV.conj(),eris.oOoV)
    fvo += einsum('kBaJ,iJkB->ai',doVvO.conj(),eris.oOoV.conj())
    fvo += einsum('aJbK,iJbK->ai',dvOvO,eris.oOvO.conj())

    fov  = einsum('ij,ja->ia',doo,eris.hov)
#    fov += 0.5 * einsum('ijkl,lkja->ia',doooo,eris.ooov)
#    fov += 0.5 * einsum('jidc,jadc->ia',doovv,eris.ovvv.conj())
#    fov +=       einsum('ibcj,jcba->ia',dovvo,eris.ovvv)
    fov += einsum('iJkL,kLaJ->ia',doOoO,eris.oOvO)
    fov += einsum('iJcD,aJcD->ia',doOvV,eris.vOvV.conj())
    fov += einsum('iBcJ,cJaB->ia',doVvO,eris.vOvV)
    fov += einsum('iBjC,jCaB->ia',doVoV,eris.oVvV)

    foo = fvv = None
    if full:
        foo  = einsum('ik,kj->ij',doo,eris.hoo)
#        foo += 0.5 * einsum('imkl,kljm->ij',doooo,eris.oooo)
#        foo += 0.5 * einsum('imab,jmab->ij',doovv,eris.oovv.conj())
#        foo +=       einsum('iabk,kbaj->ij',dovvo,eris.ovvo)
        foo += einsum('iMkL,kLjM->ij',doOoO,eris.oOoO)
        foo += einsum('iMaB,jMaB->ij',doOvV,eris.oOvV.conj())
        foo += einsum('iAbK,jAbK->ij',doVvO,eris.oVvO.conj())
        foo += einsum('iAkB,kBjA->ij',doVoV,eris.oVoV)

        fvv  = einsum('ac,cb->ab',dvv,eris.hvv)
#        fvv += 0.5 * einsum('aecd,cdbe->ab',dvvvv,eris.vvvv)
#        fvv += 0.5 * einsum('ijae,ijbe->ab',doovv.conj(),eris.oovv)
#        fvv +=       einsum('iacj,jcbi->ab',dovvo,eris.ovvo)
        fvv += einsum('aEcD,cDbE->ab',dvVvV,eris.vVvV)
        fvv += einsum('iJaE,iJbE->ab',doOvV.conj(),eris.oOvV)
        fvv += einsum('jCaI,jCbI->ab',doVvO.conj(),eris.oVvO)
        fvv += einsum('aIcJ,cJbI->ab',dvOvO,eris.vOvO)
    return foo, fov, fvo, fvv

def compute_commb_quick(d1, d2, eris, time=None, full=True):
    eris.make_tensors(time)
    doo, dvv = d1
    doooo, doovv, dovvo, dvvvv = d2
    doo, dOO = doo
    dvv, dVV = dvv
    doooo, doOoO, dOOOO = doooo
    dvvvv, dvVvV, dVVVV = dvvvv
    doovv, doOvV, dOOVV = doovv
    dovvo, doVvO, doVoV, dvOvO, dOVVO = dovvo

    fvo  = einsum('ab,ib->ai',dVV,eris.hOV.conj())
#    fvo += 0.5 * einsum('abcd,ibcd->ai',dVVVV,eris.OVVV.conj())
#    fvo += 0.5 * einsum('klab,klib->ai',dOOVV.conj(),eris.OOOV)
#    fvo +=       einsum('jabk,ijkb->ai',dOVVO,eris.OOOV.conj())
    fvo += einsum('BaDc,BiDc->ai',dvVvV,eris.vOvV.conj())
    fvo += einsum('LkBa,LkBi->ai',doOvV.conj(),eris.oOvO)
    fvo += einsum('JaBk,JiBk->ai',doVvO,eris.oOvO.conj())
    fvo += einsum('JaKb,JiKb->ai',doVoV,eris.oOoV.conj())

    fov  = einsum('ij,ja->ia',dOO,eris.hOV)
#    fov += 0.5 * einsum('ijkl,lkja->ia',dOOOO,eris.OOOV)
#    fov += 0.5 * einsum('jidc,jadc->ia',dOOVV,eris.OVVV.conj())
#    fov +=       einsum('ibcj,jcba->ia',dOVVO,eris.OVVV)
    fov += einsum('JiLk,LkJa->ia',doOoO,eris.oOoV)
    fov += einsum('JiDc,JaDc->ia',doOvV,eris.oVvV.conj())
    fov += einsum('JcBi,JcBa->ia',doVvO.conj(),eris.oVvV)
    fov += einsum('BiCj,CjBa->ia',dvOvO,eris.vOvV)

    foo = fvv = None
    if full:
        foo  = einsum('ik,kj->ij',dOO,eris.hOO)
#        foo += 0.5 * einsum('imkl,kljm->ij',dOOOO,eris.OOOO)
#        foo += 0.5 * einsum('imab,jmab->ij',dOOVV,eris.OOVV.conj())
#        foo +=       einsum('iabk,kbaj->ij',dOVVO,eris.OVVO)
        foo += einsum('MiLk,LkMj->ij',doOoO,eris.oOoO)
        foo += einsum('MiBa,MjBa->ij',doOvV,eris.oOvV.conj())
        foo += einsum('KbAi,KbAj->ij',doVvO.conj(),eris.oVvO)
        foo += einsum('AiBk,BkAj->ij',dvOvO,eris.vOvO)

        fvv  = einsum('ac,cb->ab',dVV,eris.hVV)
#        fvv += 0.5 * einsum('aecd,cdbe->ab',dVVVV,eris.VVVV)
#        fvv += 0.5 * einsum('ijae,ijbe->ab',dOOVV.conj(),eris.OOVV)
#        fvv +=       einsum('iacj,jcbi->ab',dOVVO,eris.OVVO)
        fvv += einsum('EaDc,DcEb->ab',dvVvV,eris.vVvV)
        fvv += einsum('JiEa,JiEb->ab',doOvV.conj(),eris.oOvV)
        fvv += einsum('IaCj,IbCj->ab',doVvO,eris.oVvO.conj())
        fvv += einsum('IaJc,JcIb->ab',doVoV,eris.oVoV)
    return foo, fov, fvo, fvv

def compute_X(d1, d2, eris, time):
    def compute_R(Aovvo, fov, fvo):
        no, nv = fov.shape
        Bov = fvo.T - fov.conj()
        Bov = Bov.reshape(no*nv)
        Aovvo = Aovvo.reshape(no*nv,no*nv)
        Rvo = np.dot(np.linalg.inv(Aovvo),Bov)
        Rvo = Rvo.reshape(nv,no)
        return np.block([[np.zeros((no,no)),Rvo.T.conj()],
                       [Rvo,np.zeros((nv,nv))]])
    if eris.quick:
        comma = compute_comma_quick
        commb = compute_commb_quick
    else:
        comma = compute_comma
        commb = compute_commb

    doo, dvv = d1
    Aovvo = compute_Aovvo(doo[0], dvv[0])
    _, fov, fvo, _ = comma(d1, d2, eris, time, full=False) 
    noa, nva = fov.shape
    Ra = compute_R(Aovvo, fov, fvo)
    Aovvo = compute_Aovvo(doo[1], dvv[1])
    _, fov, fvo, _ = commb(d1, d2, eris, time, full=False) 
    nob, nvb = fov.shape
    Rb = compute_R(Aovvo, fov, fvo)

    if eris.picture == 'I':
        Ra += np.block([[eris.Roo[0],np.zeros((noa,nva))],
                       [np.zeros((nva,noa)),eris.Rvv[0]]])
        Rb += np.block([[eris.Roo[1],np.zeros((nob,nvb))],
                       [np.zeros((nvb,nob)),eris.Rvv[1]]])
    return 1j*Ra, 1j*Rb

def compute_energy(d1, d2, eris, time=None):
    eris.make_tensors(time)
    doo, dvv = d1
    doooo, doovv, dovvo, dvvvv = d2
    doo, dOO = doo
    dvv, dVV = dvv
    doooo, doOoO, dOOOO = doooo
    dvvvv, dvVvV, dVVVV = dvvvv
    doovv, doOvV, dOOVV = doovv
    dovvo, doVvO, doVoV, dvOvO, dOVVO = dovvo

    e  = einsum('ij,ji',eris.hoo,doo) 
    e += einsum('ij,ji',eris.hOO,dOO) 
    e += einsum('ab,ba',eris.hvv,dvv)
    e += einsum('ab,ba',eris.hVV,dVV)
    e += 0.25 * einsum('ijkl,klij',eris.oooo,doooo) 
    e += 0.25 * einsum('ijkl,klij',eris.OOOO,dOOOO) 
    e += 0.25 * einsum('abcd,cdab',eris.vvvv,dvvvv) 
    e += 0.25 * einsum('abcd,cdab',eris.VVVV,dVVVV) 
    e += einsum('jabi,ibaj',eris.ovvo,dovvo)
    e += einsum('jabi,ibaj',eris.OVVO,dOVVO)
    tmp  = 0.25 * einsum('ijab,ijab',eris.oovv,doovv.conj())
    tmp += 0.25 * einsum('ijab,ijab',eris.OOVV,dOOVV.conj())
    tmp += tmp.conj()
    e += tmp

    e += einsum('iJkL,kLiJ',eris.oOoO,doOoO) 
    e += einsum('aBcD,cDaB',eris.vVvV,dvVvV) 
    e += einsum('jAiB,iBjA',eris.oVoV,doVoV)
    e += einsum('aJbI,bIaJ',eris.vOvO,dvOvO)
    tmp  = einsum('iJaB,iJaB',eris.oOvV,doOvV.conj())
    tmp += einsum('iBaJ,iBaJ',eris.oVvO.conj(),doVvO)
    tmp += tmp.conj()
    e += tmp
    return e.real

def compute_energy_quick(d1, d2, eris, time=None):
    eris.make_tensors(time)
    doo, dvv = d1
    doooo, doovv, dovvo, dvvvv = d2
    doo, dOO = doo
    dvv, dVV = dvv
    doooo, doOoO, dOOOO = doooo
    dvvvv, dvVvV, dVVVV = dvvvv
    doovv, doOvV, dOOVV = doovv
    dovvo, doVvO, doVoV, dvOvO, dOVVO = dovvo

    e  = einsum('ij,ji',eris.hoo,doo) 
    e += einsum('ij,ji',eris.hOO,dOO) 
    e += einsum('ab,ba',eris.hvv,dvv)
    e += einsum('ab,ba',eris.hVV,dVV)
#    e += 0.25 * einsum('ijkl,klij',eris.oooo,doooo) 
#    e += 0.25 * einsum('ijkl,klij',eris.OOOO,dOOOO) 
#    e += 0.25 * einsum('abcd,cdab',eris.vvvv,dvvvv) 
#    e += 0.25 * einsum('abcd,cdab',eris.VVVV,dVVVV) 
#    e += einsum('jabi,ibaj',eris.ovvo,dovvo)
#    e += einsum('jabi,ibaj',eris.OVVO,dOVVO)
#    tmp  = 0.25 * einsum('ijab,ijab',eris.oovv,doovv.conj())
#    tmp += 0.25 * einsum('ijab,ijab',eris.OOVV,dOOVV.conj())
#    tmp += tmp.conj()
#    e += tmp

    e += einsum('iJkL,kLiJ',eris.oOoO,doOoO) 
    e += einsum('aBcD,cDaB',eris.vVvV,dvVvV) 
    e += einsum('jAiB,iBjA',eris.oVoV,doVoV)
    e += einsum('aJbI,bIaJ',eris.vOvO,dvOvO)
    tmp  = einsum('iJaB,iJaB',eris.oOvV,doOvV.conj())
    tmp += einsum('iBaJ,iBaJ',eris.oVvO.conj(),doVvO)
    tmp += tmp.conj()
    e += tmp
    return e.real

def rotate1(h, C):
    return td_roccd_utils.rotate1(h, C)
def rotate2(eri_, Ca, Cb):
    eri = einsum('up,vq,pqrs->uvrs',Ca,Cb,eri_)
    return einsum('xr,ys,uvrs->uvxy',Ca.conj(),Cb.conj(),eri)
def fac_mol(w, td, time):
    return td_roccd_utils.fac_mol(w, td, time)
def fac_sol(sigma, w, td, time):
    return td_roccd_utils.fac_sol(sigma, w, td, time)
def phase_hubbard(A0, sigma, w, td, time):
    return td_roccd_utils.phase_hubbard(A0, sigma, w, td, time)

def mo_ints_mol(mf, z=np.zeros(3)): # field strength folded into z
    nmo = mf.mol.nao_nr()
    h0 = mf.get_hcore()
    eri = mf.mol.intor('int2e_sph')
    mu = mf.mol.intor('int1e_r')
    charges = mf.mol.atom_charges()
    coords  = mf.mol.atom_coords()
    nucl_dip = einsum('i,ix->x', charges, coords)
    if isinstance(mf.mo_coeff, np.ndarray):
        h0a = einsum('uv,up,vq->pq',h0,mf.mo_coeff,mf.mo_coeff)
        h0b = h0a.copy()

        eriab = ao2mo.full(eri, mf.mo_coeff, compact=False)
        eriab = eriab.reshape(nmo,nmo,nmo,nmo).transpose(0,2,1,3)
        eriaa = eriab - eriab.transpose(0,1,3,2)
        eribb = eriaa.copy()

        mua = einsum('xuv,up,vq->xpq',mu,mf.mo_coeff,mf.mo_coeff)
        h1a = einsum('xpq,x->pq',mua,z)
        mub, h1b = mua.copy(), h1a.copy()
    else: 
        moa, mob = mf.mo_coeff
        h0a = einsum('uv,up,vq->pq',h0,moa,moa)
        h0b = einsum('uv,up,vq->pq',h0,mob,mob)

        eriaa = ao2mo.full(eri, moa, compact=False)
        eriaa = eriaa.reshape(nmo,nmo,nmo,nmo).transpose(0,2,1,3)
        eriaa -= eriaa.transpose(0,1,3,2)
        eribb = ao2mo.full(eri, mob, compact=False)
        eribb = eribb.reshape(nmo,nmo,nmo,nmo).transpose(0,2,1,3)
        eribb -= eribb.transpose(0,1,3,2)
        eriab = ao2mo.general(eri, (moa, moa, mob, mob), compact=False)
        eriab = eriab.reshape(nmo,nmo,nmo,nmo).transpose(0,2,1,3)

        mua = einsum('xuv,up,vq->xpq',mu,moa,moa)
        mub = einsum('xuv,up,vq->xpq',mu,mob,mob)
        h1a = einsum('xpq,x->pq',mua,z)
        h1b = einsum('xpq,x->pq',mub,z)

    h0 = h0a, h0b
    h1 = h1a, h1b
    eri = eriaa, eriab, eribb
    mu = mua, mub
    return h0, h1, eri, mu, nucl_dip

def update_RK(t, l, C, eris, time, h, RK):
    if eris.quick:
        comma = compute_comma_quick
        commb = compute_commb_quick
        amps = update_amps_quick
        energy = compute_energy_quick
    else:
        comma = compute_comma
        commb = compute_commb
        amps = update_amps
        energy = compute_energy

    Fa, Fb = None, None
    Ca, Cb = C
    eris.rotate(Ca, Cb)
    d1, d2 = compute_rdm12(t, l)
    e = energy(d1, d2, eris, time=None)
    X1 = compute_X(d1, d2, eris, time)
    dt1, dl1 = amps(t, l, eris, time)
    if RK == 1:
        Fa = comma(d1, d2, eris, time)
        Fb = commb(d1, d2, eris, time)
        Fa = np.block([[Fa[0],Fa[1]],[Fa[2],Fa[3]]])
        Fb = np.block([[Fb[0],Fb[1]],[Fb[2],Fb[3]]])
        Fa -= Fa.T.conj()
        Fb -= Fb.T.conj()
        Fa = rotate1(Fa, Ca.T.conj())
        Fb = rotate1(Fb, Cb.T.conj())
        return dt1, dl1, X1, e, (Fa, Fb)
    if RK == 4:
        t_ = t[0] + dt1[0]*h*0.5, t[1] + dt1[1]*h*0.5, t[2] + dt1[2]*h*0.5 
        l_ = l[0] + dl1[0]*h*0.5, l[1] + dl1[1]*h*0.5, l[2] + dl1[2]*h*0.5 
        Ca_ = np.dot(scipy.linalg.expm(-0.5*h*X1[0]), Ca)
        Cb_ = np.dot(scipy.linalg.expm(-0.5*h*X1[1]), Cb)
        eris.rotate(Ca_, Cb_)
        d1, d2 = compute_rdm12(t_, l_)
        X2 = compute_X(d1, d2, eris, time+h*0.5)
        dt2, dl2 = amps(t_, l_, eris, time+h*0.5) 

        t_ = t[0] + dt2[0]*h*0.5, t[1] + dt2[1]*h*0.5, t[2] + dt2[2]*h*0.5 
        l_ = l[0] + dl2[0]*h*0.5, l[1] + dl2[1]*h*0.5, l[2] + dl2[2]*h*0.5 
        Ca_ = np.dot(scipy.linalg.expm(-0.5*h*X2[0]), Ca)
        Cb_ = np.dot(scipy.linalg.expm(-0.5*h*X2[1]), Cb)
        eris.rotate(Ca_, Cb_)
        d1, d2 = compute_rdm12(t_, l_)
        X3 = compute_X(d1, d2, eris, time+h*0.5)
        dt3, dl3 = amps(t_, l_, eris, time+h*0.5)
 
        t_ = t[0] + dt3[0]*h, t[1] + dt3[1]*h, t[2] + dt3[2]*h 
        l_ = l[0] + dl3[0]*h, l[1] + dl3[1]*h, l[2] + dl3[2]*h 
        Ca_ = np.dot(scipy.linalg.expm(-h*X2[0]), Ca)
        Cb_ = np.dot(scipy.linalg.expm(-h*X2[1]), Cb)
        eris.rotate(Ca_, Cb_)
        d1, d2 = compute_rdm12(t_, l_)
        X4 = compute_X(d1, d2, eris, time+h)
        dt4, dl4 = amps(t_, l_, eris, time+h) 

        dtaa = (dt1[0] + 2.0*dt2[0] + 2.0*dt3[0] + dt4[0])/6.0
        dtab = (dt1[1] + 2.0*dt2[1] + 2.0*dt3[1] + dt4[1])/6.0
        dtbb = (dt1[2] + 2.0*dt2[2] + 2.0*dt3[2] + dt4[2])/6.0
        dlaa = (dl1[0] + 2.0*dl2[0] + 2.0*dl3[0] + dl4[0])/6.0
        dlab = (dl1[1] + 2.0*dl2[1] + 2.0*dl3[1] + dl4[1])/6.0
        dlbb = (dl1[2] + 2.0*dl2[2] + 2.0*dl3[2] + dl4[2])/6.0
        Xa = (X1[0] + 2.0*X2[0] + 2.0*X3[0] + X4[0])/6.0
        Xb = (X1[1] + 2.0*X2[1] + 2.0*X3[1] + X4[1])/6.0
        dt, dl, X = (dtaa, dtab, dtbb), (dlaa, dlab, dlbb), (Xa, Xb)
        return dt, dl, X, e, (Fa, Fb)

def compute_sqrt_fd(mo_energy, beta, mu):
    if isinstance(mo_energy, np.ndarray):
        fda = td_roccd_utils.compute_sqrt_fd(mo_energy, beta, mu)
        fdb = fda[0].copy(), fda[1].copy()
    else:
        fda = td_roccd_utils.compute_sqrt_fd(mo_energy[0], beta, mu)
        fdb = td_roccd_utils.compute_sqrt_fd(mo_energy[1], beta, mu)
    return fda, fdb

def make_bogoliubov1(h, fd, fd_):
    return td_roccd_utils.make_bogoliubov1(h, fd, fd_)
def make_bogoliubov2(eri, fda, fdb, fda_, fdb_):
    no = len(fda)
    erib = np.zeros((no*2,)*4, dtype=complex)
    erib[:no,:no,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fda ,fdb ,fda ,fdb )
    erib[no:,:no,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fda_,fdb ,fda ,fdb )
    erib[:no,no:,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fda ,fdb_,fda ,fdb )
    erib[:no,:no,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fda ,fdb ,fda_,fdb )
    erib[:no,:no,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fda ,fdb ,fda ,fdb_)
    erib[:no,:no,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fda ,fdb ,fda_,fdb_)
    erib[no:,no:,:no,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fda_,fdb_,fda ,fdb )
    erib[:no,no:,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fda ,fdb_,fda_,fdb )
    erib[:no,no:,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fda ,fdb_,fda ,fdb_)
    erib[no:,:no,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fda_,fdb ,fda ,fdb_)
    erib[no:,:no,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fda_,fdb ,fda_,fdb )
    erib[:no,no:,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fda ,fdb_,fda_,fdb_)
    erib[no:,:no,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fda_,fdb ,fda_,fdb_)
    erib[no:,no:,:no,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fda_,fdb_,fda ,fdb_)
    erib[no:,no:,no:,:no] = einsum('pqrs,p,q,r,s->pqrs',eri,fda_,fdb_,fda_,fdb )
    erib[no:,no:,no:,no:] = einsum('pqrs,p,q,r,s->pqrs',eri,fda_,fdb_,fda_,fdb_)
    return erib

def make_Roo(mo_energy, fda, fdb):
    if isinstance(mo_energy, np.ndarray):
        Roo = td_roccd_utils.make_Roo(mo_energy, fda)
        ROO = Roo.copy()
    else: 
        Roo = td_roccd_utils.make_Roo(mo_energy[0], fda)
        ROO = td_roccd_utils.make_Roo(mo_energy[1], fdb)
    return Roo, ROO

def build_rdm1(d1):
    doo, dvv = d1
    doo, dOO = doo
    dvv, dVV = dvv
    noa, nob = doo.shape[0], dOO.shape[1]
    nva, nvb = dvv.shape[0], dVV.shape[1]
    d1a = np.block([[doo,np.zeros((noa,nva))],
                    [np.zeros((nva,noa)),dvv]])
    d1b = np.block([[dOO,np.zeros((nob,nvb))],
                    [np.zeros((nvb,nob)),dVV]])
    return d1a, d1b

def build_rdm2ab(d2):
    doooo, doovv, dovvo, dvvvv = d2
    doooo, doOoO, dOOOO = doooo 
    dvvvv, dvVvV, dVVVV = dvvvv 
    doovv, doOvV, dOOVV = doovv  
    dovvo, doVvO, doVoV, dvOvO, dOVVO = dovvo
    noa, nob, nva, nvb = doOvV.shape
    nmoa, nmob = noa + nva, nob + nvb
    d2ab = np.zeros((nmoa,nmob,nmoa,nmob),dtype=complex)
    d2ab[:noa,:nob,:noa,:nob] = doOoO.copy()
    d2ab[noa:,nob:,noa:,nob:] = dvVvV.copy()
    d2ab[:noa,:nob,noa:,nob:] = doOvV.copy()
    d2ab[noa:,nob:,:noa,:nob] = doOvV.transpose(2,3,0,1).conj().copy()
    d2ab[:noa,nob:,noa:,:nob] = doVvO.copy()
    d2ab[noa:,:nob,:noa,nob:] = doVvO.transpose(2,3,0,1).conj().copy()
    d2ab[:noa,nob:,:noa,nob:] = doVoV.copy()
    d2ab[noa:,:nob,noa:,:nob] = dvOvO.copy()
    return d2ab

def compute_phys1(d1, fd, fd_):
    return td_roccd_utils.compute_phys1(d1, fd, fd_)

def compute_phys2(d2b, fda, fdb, fda_, fdb_):
    no = len(fda)
    d2  = einsum('pqrs,p,q,r,s->pqrs',d2b[:no,:no,:no,:no],fda ,fdb ,fda ,fdb )
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[no:,:no,:no,:no],fda_,fdb ,fda ,fdb )
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[:no,no:,:no,:no],fda ,fdb_,fda ,fdb )
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[:no,:no,no:,:no],fda ,fdb ,fda_,fdb )
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[:no,:no,:no,no:],fda ,fdb ,fda ,fdb_)
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[:no,:no,no:,no:],fda ,fdb ,fda_,fdb_)
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[no:,no:,:no,:no],fda_,fdb_,fda ,fdb )
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[:no,no:,no:,:no],fda ,fdb_,fda_,fdb )
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[:no,no:,:no,no:],fda ,fdb_,fda ,fdb_)
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[no:,:no,:no,no:],fda_,fdb ,fda ,fdb_)
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[no:,:no,no:,:no],fda_,fdb ,fda_,fdb )
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[:no,no:,no:,no:],fda ,fdb_,fda_,fdb_)
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[no:,:no,no:,no:],fda_,fdb ,fda_,fdb_)
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[no:,no:,:no,no:],fda_,fdb_,fda ,fdb_)
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[no:,no:,no:,:no],fda_,fdb_,fda_,fdb )
    d2 += einsum('pqrs,p,q,r,s->pqrs',d2b[no:,no:,no:,no:],fda_,fdb_,fda_,fdb_)
    return d2

def tdhf(h, V, nelec=None, Pa=None, Pb=None, beta=None, mu=None, thresh=1e-8):
    Va = V - V.transpose(0,1,3,2)
    def fd(mo_energy):
        if beta is None:
            na = np.zeros(h.shape[0])
            nb = np.zeros(h.shape[0])
            na[:nelec[0]] = 1.0
            nb[:nelec[1]] = 1.0
        else:
            fda, fdb = compute_sqrt_fd(mo_energy, beta, mu)
            na, nb = np.square(fda[0]), np.square(fdb[0])
        return na, nb 
    def compute_mo(Pa, Pb):
        Fa, Fb = h.copy(), h.copy()
        Fa += einsum('pqrs,qs->pr',Va,Pa)
        Fa += einsum('pqrs,qs->pr',V ,Pb)
        Fb += einsum('pqrs,qs->pr',Va,Pb)
        Fb += einsum('pqrs,pr->qs',V ,Pa)
        ea, ua = np.linalg.eigh(Fa)
        eb, ub = np.linalg.eigh(Fb)
        return (ua, ub), (ea, eb)
    def compute_fock(mo_coeff, mo_energy):
        ua, ub = mo_coeff
        na, nb = fd(mo_energy) # in mo basis
        Fa, Fb = rotate1(h, ua.T), rotate1(h, ub.T) # in mo basis
        eriaa = rotate2(Va, ua.T, ua.T)
        eribb = rotate2(Va, ub.T, ub.T)
        eriab = rotate2(V , ua.T, ub.T)
        Fa += einsum('pqrq,q->pr',eriaa,na)
        Fa += einsum('pqrq,q->pr',eriab,nb)
        Fb += einsum('pqrq,q->pr',eribb,nb)
        Fb += einsum('pqps,p->qs',eriab,na)
        return Fa, Fb
    conv = False
    maxiter = 100
    if Pa is None:
        e, u = np.linalg.eigh(h)
        na, nb = fd(e)
        Pa = einsum('pr,qr,r->pq',u,u,na)
        Pb = einsum('pr,qr,r->pq',u,u,nb) 
    for i in range(maxiter):
        mo_coeff, mo_energy = compute_mo(Pa, Pb)
#        print(np.linalg.norm(mo_coeff[0]-mo_coeff[1]))
        Fa, Fb = compute_fock(mo_coeff, mo_energy) # in mo basis
        Fa, Fb = rotate1(Fa, mo_coeff[0]), rotate1(Fb, mo_coeff[1]) # in ao basis
        ea, ua = np.linalg.eigh(Fa)
        eb, ub = np.linalg.eigh(Fb)
        na, nb = fd((ea, eb))
        Panew = einsum('pr,qr,r->pq',ua,ua,na)
        Pbnew = einsum('pr,qr,r->pq',ub,ub,nb)
        dnorm  = np.linalg.norm(Panew-Pa)
#        dnorm += np.linalg.norm(Pbnew-Pb)
#        print(i, dnorm)
        Pa, Pb = Panew, Pbnew
        if dnorm < thresh:
            break
    return (Pa, Pb), mo_coeff, mo_energy 

#def getH_sector(h, V, model, nelec, ms=None): 
#    myfci = FCISimple(model, nelec, ms)
#    k, nelec = myfci.basis.shape
#    assert(k==myfci.k)
#    assert(nelec==myfci.nelec)
#    H = np.zeros((k,k),dtype=complex)
#    for i in range(k):
#        for j in range(k):
#            istate = myfci.basis[i,:]
#            jstate = myfci.basis[j,:]
#            H[i,j] = myfci._get_matrixel(istate,jstate,V,h)
#    return H

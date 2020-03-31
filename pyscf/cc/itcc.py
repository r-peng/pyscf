import time, math, scipy
import numpy as np

from pyscf import lib, ao2mo, __config__
from pyscf.lib import logger
from pyscf.cc import ccsd, uccsd
from pyscf.fci.cistring import _gen_occslst

np.set_printoptions(precision=8,suppress=True)
einsum = lib.einsum
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

make_tau = uccsd.make_tau
make_tau_aa = uccsd.make_tau_aa
make_tau_ab = uccsd.make_tau_ab

def get_frozen(mf, frozen):
    nea, neb = mf.mol.nelec
    if type(frozen) is int:
        nfc = frozen, frozen
        nfv = 0,0
    if type(frozen) is list:
        nfc = len(frozen[frozen<nea]),len(frozen[frozen<neb])
        nfv = len(frozen[frozen>nea]),len(frozen[frozen>neb])
    if type(frozen) is tuple:
        nfc = len(frozen[0][frozen[0]<nea]),len(frozen[1][frozen[1]<neb])
        nfv = len(frozen[0][frozen[0]>nea]),len(frozen[1][frozen[1]>neb])
    return nfc, nfv

def get_occ(mf, ncas, nelecas, occslst, ci, ndet, cutoff):
    if occslst is None:
        if ci is None: return cas2occ(mf, ncas, nelecas)
        else: return ci2occ(mf, ci, ndet, cutoff)
    else: return occslst
        
def ci2occ(mf, ci, ndet=None, cutoff=None, ne=None, norb=None):
    nea, neb = mf.mol.nelec if ne is None else ne
    norba = mf.mo_coeff[0].shape[1] if norb is None else norb[0]
    norbb = mf.mo_coeff[1].shape[1] if norb is None else norb[1]
    nca, ncb = mf.mol.nelec[0]-nea, mf.mol.nelec[1]-neb
    occslsta = _gen_occslst(range(nca,nca+norba),nea)
    occslstb = _gen_occslst(range(ncb,ncb+norbb),neb)
    Na, Nb = occslsta.shape[0], occslstb.shape[0]
    assert ci.shape == (Na,Nb)
    ci = np.ravel(ci)
    if ndet is None:
        ndet = ci[abs(ci)>cutoff].size
    idx = np.argsort(abs(ci))[-ndet:]
    idx = np.flip(idx)
    occ = np.zeros((ndet,nea+neb),dtype=int)
    for I in range(ndet):
        Ia, Ib = np.divmod(idx[I],Na)
        occ[I,:nea] = occslsta[Ia,:]
        occ[I,nea:] = occslstb[Ib,:]
    print('important determinants:\n{}'.format(occ))
    print('coeffs for important determinants:\n{}'.format(ci[idx]))
    print('ci vector size: {}'.format(Na*Nb))
    return occ

def cas2occ(mf, ncas, nelecas):
    nea, neb = mf.mol.nelec 
    nca = nea-nelecas[0]
    ncb = neb-nelecas[1]
    occslsta = _gen_occslst(range(nca,nca+ncas[0]),nelecas[0])
    occslstb = _gen_occslst(range(ncb,ncb+ncas[1]),nelecas[1])
    ca = np.empty((occslsta.shape[0],nca),dtype=int)
    cb = np.empty((occslstb.shape[0],ncb),dtype=int)
    for i in range(nca): 
        ca[:, i] = i
    for i in range(ncb): 
        cb[:, i] = i
    occslsta = np.hstack((ca, occslsta))
    occslstb = np.hstack((cb, occslstb))
    Na = occslsta.shape[0]
    Nb = occslstb.shape[0]
    ndet = Na*Nb
    occ = np.empty((ndet,nea+neb),dtype=int)
    for Ia in range(Na):
        for Ib in range(Nb):
            I = Ia*Na + Ib
            occ[I,:] = np.hstack((occslsta[Ia,:],occslstb[Ib,:]))
    print('number of active determinant: {}'.format(ndet))
    return occ 

def perm_mo(mf, mo0, occ):
    nea, neb = mf.mol.nelec
    moa, mob = mo0
    nmoa, nmob = moa.shape[1], mob.shape[1]
    occa, occb = occ[:nea], occ[nea:]
    vira = list(set(range(nmoa))-set(occa)) 
    virb = list(set(range(nmoa))-set(occb))
    moa_occ, mob_occ = moa[:,list(occa)], mob[:,list(occb)]
    moa_vir, mob_vir = moa[:,vira], mob[:,virb]
    vira, virb = np.array(vira), np.array(virb)
    moa, mob = np.hstack((moa_occ,moa_vir)), np.hstack((mob_occ,mob_vir))
    occ = np.hstack((occa,occb))
    vir = np.hstack((vira,virb))
    return (moa, mob), occ, vir

def ref_energy(mf):
    moa, mob = mf.mo_coeff
    nmoa, nmob = moa.shape[1], mob.shape[1]
    nea, neb = mf.mol.nelec
    hcore = mf.get_hcore()
    ha = np.linalg.multi_dot([moa.T,hcore,moa])
    hb = np.linalg.multi_dot([mob.T,hcore,mob])
    E0 = sum(ha.diagonal()[:nea]) + sum(hb.diagonal()[:neb])

    eriao = mf._eri
    eri_aa = ao2mo.restore(1, ao2mo.full(eriao, moa), nmoa)
    eri_bb = ao2mo.restore(1, ao2mo.full(eriao, mob), nmob)
    eri_ab = ao2mo.general(eriao, (moa,moa,mob,mob), compact=False)
    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    oooo = eri_aa[:nea,:nea,:nea,:nea].copy()
    OOOO = eri_bb[:neb,:neb,:neb,:neb].copy()
    ooOO = eri_ab[:nea,:nea,:neb,:neb].copy()

    temp = einsum('iijj',OOOO)-einsum('ijji',OOOO)
    temp += einsum('iijj',oooo)-einsum('ijji',oooo) 
    temp += 2*einsum('iijj',ooOO)
    E0 += 0.5*temp
    return E0 

def update_cc(cc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]

    u1a = np.zeros_like(t1a)
    u1b = np.zeros_like(t1b)
    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:u2aa += lib.einsum('ijef,aebf->ijab', tauaa, eris_vvvv) * .5
    #:u2bb += lib.einsum('ijef,aebf->ijab', taubb, eris_VVVV) * .5
    #:u2ab += lib.einsum('iJeF,aeBF->iJaB', tauab, eris_vvVV)
    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    u2aa, u2ab, u2bb = cc._add_vvvv(None, (tauaa,tauab,taubb), eris)
    u2aa *= .5
    u2bb *= .5

    Fooa =  .5 * lib.einsum('me,ie->mi', fova, t1a)
    Foob =  .5 * lib.einsum('me,ie->mi', fovb, t1b)
    Fvva = -.5 * lib.einsum('me,ma->ae', fova, t1a)
    Fvvb = -.5 * lib.einsum('me,ma->ae', fovb, t1b)
    Fooa += eris.focka[:nocca,:nocca]
    Foob += eris.fockb[:noccb,:noccb]
    Fvva += eris.focka[nocca:,nocca:]
    Fvvb += eris.fockb[noccb:,noccb:]
    dtype = u2aa.dtype
    wovvo = np.zeros((nocca,nvira,nvira,nocca), dtype=dtype)
    wOVVO = np.zeros((noccb,nvirb,nvirb,noccb), dtype=dtype)
    woVvO = np.zeros((nocca,nvirb,nvira,noccb), dtype=dtype)
    woVVo = np.zeros((nocca,nvirb,nvirb,nocca), dtype=dtype)
    wOvVo = np.zeros((noccb,nvira,nvirb,nocca), dtype=dtype)
    wOvvO = np.zeros((noccb,nvira,nvira,noccb), dtype=dtype)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now - u2aa.size*8e-6)
    if nvira > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)
            wovvo[p0:p1] += lib.einsum('jf,mebf->mbej', t1a, ovvv)
            u1a += 0.5*lib.einsum('mief,meaf->ia', t2aa[p0:p1], ovvv)
            u2aa[:,p0:p1] += lib.einsum('ie,mbea->imab', t1a, ovvv.conj())
            tmp1aa = lib.einsum('ijef,mebf->ijmb', tauaa, ovvv)
            u2aa -= lib.einsum('ijmb,ma->ijab', tmp1aa, t1a[p0:p1]*.5)
            ovvv = tmp1aa = None

    if nvirb > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)
            wOVVO[p0:p1] = lib.einsum('jf,mebf->mbej', t1b, OVVV)
            u1b += 0.5*lib.einsum('MIEF,MEAF->IA', t2bb[p0:p1], OVVV)
            u2bb[:,p0:p1] += lib.einsum('ie,mbea->imab', t1b, OVVV.conj())
            tmp1bb = lib.einsum('ijef,mebf->ijmb', taubb, OVVV)
            u2bb -= lib.einsum('ijmb,ma->ijab', tmp1bb, t1b[p0:p1]*.5)
            OVVV = tmp1bb = None

    if nvirb > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)
            woVvO[p0:p1] = lib.einsum('JF,meBF->mBeJ', t1b, ovVV)
            woVVo[p0:p1] = lib.einsum('jf,mfBE->mBEj',-t1a, ovVV)
            u1b += lib.einsum('mIeF,meAF->IA', t2ab[p0:p1], ovVV)
            u2ab[p0:p1] += lib.einsum('IE,maEB->mIaB', t1b, ovVV.conj())
            tmp1ab = lib.einsum('iJeF,meBF->iJmB', tauab, ovVV)
            u2ab -= lib.einsum('iJmB,ma->iJaB', tmp1ab, t1a[p0:p1])
            ovVV = tmp1ab = None

    if nvira > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)
            wOvVo[p0:p1] = lib.einsum('jf,MEbf->MbEj', t1a, OVvv)
            wOvvO[p0:p1] = lib.einsum('JF,MFbe->MbeJ',-t1b, OVvv)
            u1a += lib.einsum('iMfE,MEaf->ia', t2ab[:,p0:p1], OVvv)
            u2ab[:,p0:p1] += lib.einsum('ie,MBea->iMaB', t1a, OVvv.conj())
            tmp1abba = lib.einsum('iJeF,MFbe->iJbM', tauab, OVvv)
            u2ab -= lib.einsum('iJbM,MA->iJbA', tmp1abba, t1b[p0:p1])
            OVvv = tmp1abba = None

    eris_ovov = np.asarray(eris.ovov)
    eris_ovoo = np.asarray(eris.ovoo)
    Woooo = lib.einsum('je,nemi->mnij', t1a, eris_ovoo)
    Woooo = Woooo - Woooo.transpose(0,1,3,2)
    Woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
    Woooo += lib.einsum('ijef,menf->mnij', tauaa, eris_ovov) * .5
    u2aa += lib.einsum('mnab,mnij->ijab', tauaa, Woooo*.5)
    Woooo = tauaa = None
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    Fooa += np.einsum('ne,nemi->mi', t1a, ovoo)
    u1a += 0.5*lib.einsum('mnae,meni->ia', t2aa, ovoo)
    wovvo += lib.einsum('nb,nemj->mbej', t1a, ovoo)
    ovoo = eris_ovoo = None

    tilaa = make_tau_aa(t2[0], t1a, t1a, fac=0.5)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    Fvva -= .5 * lib.einsum('mnaf,menf->ae', tilaa, ovov)
    Fooa += .5 * lib.einsum('inef,menf->mi', tilaa, ovov)
    Fova = np.einsum('nf,menf->me',t1a, ovov)
    u2aa += ovov.conj().transpose(0,2,1,3) * .5
    wovvo -= 0.5*lib.einsum('jnfb,menf->mbej', t2aa, ovov)
    woVvO += 0.5*lib.einsum('nJfB,menf->mBeJ', t2ab, ovov)
    tmpaa = lib.einsum('jf,menf->mnej', t1a, ovov)
    wovvo -= lib.einsum('nb,mnej->mbej', t1a, tmpaa)
    eirs_ovov = ovov = tmpaa = tilaa = None

    eris_OVOV = np.asarray(eris.OVOV)
    eris_OVOO = np.asarray(eris.OVOO)
    WOOOO = lib.einsum('je,nemi->mnij', t1b, eris_OVOO)
    WOOOO = WOOOO - WOOOO.transpose(0,1,3,2)
    WOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
    WOOOO += lib.einsum('ijef,menf->mnij', taubb, eris_OVOV) * .5
    u2bb += lib.einsum('mnab,mnij->ijab', taubb, WOOOO*.5)
    WOOOO = taubb = None
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    Foob += np.einsum('ne,nemi->mi', t1b, OVOO)
    u1b += 0.5*lib.einsum('mnae,meni->ia', t2bb, OVOO)
    wOVVO += lib.einsum('nb,nemj->mbej', t1b, OVOO)
    OVOO = eris_OVOO = None

    tilbb = make_tau_aa(t2[2], t1b, t1b, fac=0.5)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Fvvb -= .5 * lib.einsum('MNAF,MENF->AE', tilbb, OVOV)
    Foob += .5 * lib.einsum('inef,menf->mi', tilbb, OVOV)
    Fovb = np.einsum('nf,menf->me',t1b, OVOV)
    u2bb += OVOV.conj().transpose(0,2,1,3) * .5
    wOVVO -= 0.5*lib.einsum('jnfb,menf->mbej', t2bb, OVOV)
    wOvVo += 0.5*lib.einsum('jNbF,MENF->MbEj', t2ab, OVOV)
    tmpbb = lib.einsum('jf,menf->mnej', t1b, OVOV)
    wOVVO -= lib.einsum('nb,mnej->mbej', t1b, tmpbb)
    eris_OVOV = OVOV = tmpbb = tilbb = None

    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    Fooa += np.einsum('NE,NEmi->mi', t1b, eris_OVoo)
    u1a -= lib.einsum('nMaE,MEni->ia', t2ab, eris_OVoo)
    wOvVo -= lib.einsum('nb,MEnj->MbEj', t1a, eris_OVoo)
    woVVo += lib.einsum('NB,NEmj->mBEj', t1b, eris_OVoo)
    Foob += np.einsum('ne,neMI->MI', t1a, eris_ovOO)
    u1b -= lib.einsum('mNeA,meNI->IA', t2ab, eris_ovOO)
    woVvO -= lib.einsum('NB,meNJ->mBeJ', t1b, eris_ovOO)
    wOvvO += lib.einsum('nb,neMJ->MbeJ', t1a, eris_ovOO)
    WoOoO = lib.einsum('JE,NEmi->mNiJ', t1b, eris_OVoo)
    WoOoO+= lib.einsum('je,neMI->nMjI', t1a, eris_ovOO)
    WoOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)
    eris_OVoo = eris_ovOO = None

    eris_ovOV = np.asarray(eris.ovOV)
    WoOoO += lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)
    u2ab += lib.einsum('mNaB,mNiJ->iJaB', tauab, WoOoO)
    WoOoO = None

    tilab = make_tau_ab(t2[1], t1 , t1 , fac=0.5)
    Fvva -= lib.einsum('mNaF,meNF->ae', tilab, eris_ovOV)
    Fvvb -= lib.einsum('nMfA,nfME->AE', tilab, eris_ovOV)
    Fooa += lib.einsum('iNeF,meNF->mi', tilab, eris_ovOV)
    Foob += lib.einsum('nIfE,nfME->MI', tilab, eris_ovOV)
    Fova += np.einsum('NF,meNF->me',t1b, eris_ovOV)
    Fovb += np.einsum('nf,nfME->ME',t1a, eris_ovOV)
    u2ab += eris_ovOV.conj().transpose(0,2,1,3)
    wovvo += 0.5*lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
    wOVVO += 0.5*lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
    wOvVo -= 0.5*lib.einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
    woVvO -= 0.5*lib.einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
    woVVo += 0.5*lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
    wOvvO += 0.5*lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
    tmpabab = lib.einsum('JF,meNF->mNeJ', t1b, eris_ovOV)
    tmpbaba = lib.einsum('jf,nfME->MnEj', t1a, eris_ovOV)
    woVvO -= lib.einsum('NB,mNeJ->mBeJ', t1b, tmpabab)
    wOvVo -= lib.einsum('nb,MnEj->MbEj', t1a, tmpbaba)
    woVVo += lib.einsum('NB,NmEj->mBEj', t1b, tmpbaba)
    wOvvO += lib.einsum('nb,nMeJ->MbeJ', t1a, tmpabab)
    tmpabab = tmpbaba = tilab = None

    Fova += fova
    Fovb += fovb
    u1a += fova.conj()
    u1a += np.einsum('ie,ae->ia', t1a, Fvva)
    u1a -= np.einsum('ma,mi->ia', t1a, Fooa)
    u1a -= np.einsum('imea,me->ia', t2aa, Fova)
    u1a += np.einsum('iMaE,ME->ia', t2ab, Fovb)
    u1b += fovb.conj()
    u1b += np.einsum('ie,ae->ia',t1b,Fvvb)
    u1b -= np.einsum('ma,mi->ia',t1b,Foob)
    u1b -= np.einsum('imea,me->ia', t2bb, Fovb)
    u1b += np.einsum('mIeA,me->IA', t2ab, Fova)

    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    wovvo -= eris_oovv.transpose(0,2,3,1)
    wovvo += eris_ovvo.transpose(0,2,1,3)
    oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
    u1a-= np.einsum('nf,niaf->ia', t1a,      oovv)
    tmp1aa = lib.einsum('ie,mjbe->mbij', t1a,      oovv)
    u2aa += 2*lib.einsum('ma,mbij->ijab', t1a, tmp1aa)
    eris_ovvo = eris_oovv = oovv = tmp1aa = None

    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    wOVVO -= eris_OOVV.transpose(0,2,3,1)
    wOVVO += eris_OVVO.transpose(0,2,1,3)
    OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
    u1b-= np.einsum('nf,niaf->ia', t1b,      OOVV)
    tmp1bb = lib.einsum('ie,mjbe->mbij', t1b,      OOVV)
    u2bb += 2*lib.einsum('ma,mbij->ijab', t1b, tmp1bb)
    eris_OVVO = eris_OOVV = OOVV = None

    eris_ooVV = np.asarray(eris.ooVV)
    eris_ovVO = np.asarray(eris.ovVO)
    woVVo -= eris_ooVV.transpose(0,2,3,1)
    woVvO += eris_ovVO.transpose(0,2,1,3)
    u1b+= np.einsum('nf,nfAI->IA', t1a, eris_ovVO)
    tmp1ab = lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
    tmp1ab+= lib.einsum('IE,mjBE->mBjI', t1b, eris_ooVV)
    u2ab -= lib.einsum('ma,mBiJ->iJaB', t1a, tmp1ab)
    eris_ooVV = eris_ovVo = tmp1ab = None

    eris_OOvv = np.asarray(eris.OOvv)
    eris_OVvo = np.asarray(eris.OVvo)
    wOvvO -= eris_OOvv.transpose(0,2,3,1)
    wOvVo += eris_OVvo.transpose(0,2,1,3)
    u1a+= np.einsum('NF,NFai->ia', t1b, eris_OVvo)
    tmp1ba = lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
    tmp1ba+= lib.einsum('ie,MJbe->MbJi', t1a, eris_OOvv)
    u2ab -= lib.einsum('MA,MbIj->jIbA', t1b, tmp1ba)
    eris_OOvv = eris_OVvO = tmp1ba = None

    u2aa += 2*lib.einsum('imae,mbej->ijab', t2aa, wovvo)
    u2aa += 2*lib.einsum('iMaE,MbEj->ijab', t2ab, wOvVo)
    u2bb += 2*lib.einsum('imae,mbej->ijab', t2bb, wOVVO)
    u2bb += 2*lib.einsum('mIeA,mBeJ->IJAB', t2ab, woVvO)
    u2ab += lib.einsum('imae,mBeJ->iJaB', t2aa, woVvO)
    u2ab += lib.einsum('iMaE,MBEJ->iJaB', t2ab, wOVVO)
    u2ab += lib.einsum('iMeA,MbeJ->iJbA', t2ab, wOvvO)
    u2ab += lib.einsum('IMAE,MbEj->jIbA', t2bb, wOvVo)
    u2ab += lib.einsum('mIeA,mbej->jIbA', t2ab, wovvo)
    u2ab += lib.einsum('mIaE,mBEj->jIaB', t2ab, woVVo)
    wovvo = wOVVO = woVvO = wOvVo = woVVo = wOvvO = None

    Ftmpa = Fvva - .5*lib.einsum('mb,me->be', t1a, Fova)
    Ftmpb = Fvvb - .5*lib.einsum('mb,me->be', t1b, Fovb)
    u2aa += lib.einsum('ijae,be->ijab', t2aa, Ftmpa)
    u2bb += lib.einsum('ijae,be->ijab', t2bb, Ftmpb)
    u2ab += lib.einsum('iJaE,BE->iJaB', t2ab, Ftmpb)
    u2ab += lib.einsum('iJeA,be->iJbA', t2ab, Ftmpa)
    Ftmpa = Fooa + 0.5*lib.einsum('je,me->mj', t1a, Fova)
    Ftmpb = Foob + 0.5*lib.einsum('je,me->mj', t1b, Fovb)
    u2aa -= lib.einsum('imab,mj->ijab', t2aa, Ftmpa)
    u2bb -= lib.einsum('imab,mj->ijab', t2bb, Ftmpb)
    u2ab -= lib.einsum('iMaB,MJ->iJaB', t2ab, Ftmpb)
    u2ab -= lib.einsum('mIaB,mj->jIaB', t2ab, Ftmpa)

    eris_ovoo = np.asarray(eris.ovoo).conj()
    eris_OVOO = np.asarray(eris.OVOO).conj()
    eris_OVoo = np.asarray(eris.OVoo).conj()
    eris_ovOO = np.asarray(eris.ovOO).conj()
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    u2aa -= lib.einsum('ma,jbim->ijab', t1a, ovoo)
    u2bb -= lib.einsum('ma,jbim->ijab', t1b, OVOO)
    u2ab -= lib.einsum('ma,JBim->iJaB', t1a, eris_OVoo)
    u2ab -= lib.einsum('MA,ibJM->iJbA', t1b, eris_ovOO)
    eris_ovoo = eris_OVoo = eris_OVOO = eris_ovOO = None

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)

    time0 = log.timer_debug1('update t1 t2', *time0)
    t1new = -u1a, -u1b
    t2new = -u2aa, -u2ab, -u2bb
    return t1new, t2new

def t2ci(t1, t2):
    t1a, t1b = t1
    c2aa, c2ab, c2bb = t2
    c2aa += einsum('ia,jb->ijab',t1a,t1a) - einsum('ib,ja->ijab',t1a,t1a) 
    c2bb += einsum('ia,jb->ijab',t1b,t1b) - einsum('ib,ja->ijab',t1b,t1b)
    c2ab += einsum('ia,jb->ijab',t1a,t1b)
    return t1, (c2aa, c2ab, c2bb)  

def ci2t(c1, c2):
    t1a, t1b = c1
    t2aa, t2ab, t2bb = c2
    t2aa -= einsum('ia,jb->ijab',t1a,t1a) - einsum('ib,ja->ijab',t1a,t1a) 
    t2bb -= einsum('ia,jb->ijab',t1b,t1b) - einsum('ib,ja->ijab',t1b,t1b)
    t2ab -= einsum('ia,jb->ijab',t1a,t1b)
    return c1, (t2aa, t2ab, t2bb)  

def energy_ci(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    e  = np.einsum('ia,ia', fova, t1a)
    e += np.einsum('ia,ia', fovb, t1b)
    e += 0.25*np.einsum('ijab,iajb',t2aa,eris_ovov)
    e -= 0.25*np.einsum('ijab,ibja',t2aa,eris_ovov)
    e += 0.25*np.einsum('ijab,iajb',t2bb,eris_OVOV)
    e -= 0.25*np.einsum('ijab,ibja',t2bb,eris_OVOV)
    e +=      np.einsum('iJaB,iaJB',t2ab,eris_ovOV)
    return e.real

def update_ci(cc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]

    u1a = np.zeros_like(t1a)
    u1b = np.zeros_like(t1b)
    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:u2aa += lib.einsum('ijef,aebf->ijab', tauaa, eris_vvvv) * .5
    #:u2bb += lib.einsum('ijef,aebf->ijab', taubb, eris_VVVV) * .5
    #:u2ab += lib.einsum('iJeF,aeBF->iJaB', tauab, eris_vvVV)
    tauaa, tauab, taubb = t2
    u2aa, u2ab, u2bb = cc._add_vvvv(None, (tauaa,tauab,taubb), eris)
    u2aa *= .5
    u2bb *= .5

    Fooa = eris.focka[:nocca,:nocca]
    Foob = eris.fockb[:noccb,:noccb]
    Fvva = eris.focka[nocca:,nocca:]
    Fvvb = eris.fockb[noccb:,noccb:]
    dtype = u2aa.dtype
    wovvo = np.zeros((nocca,nvira,nvira,nocca), dtype=dtype)
    wOVVO = np.zeros((noccb,nvirb,nvirb,noccb), dtype=dtype)
    woVvO = np.zeros((nocca,nvirb,nvira,noccb), dtype=dtype)
    woVVo = np.zeros((nocca,nvirb,nvirb,nocca), dtype=dtype)
    wOvVo = np.zeros((noccb,nvira,nvirb,nocca), dtype=dtype)
    wOvvO = np.zeros((noccb,nvira,nvira,noccb), dtype=dtype)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now - u2aa.size*8e-6)
    if nvira > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            u1a += 0.5*lib.einsum('mief,meaf->ia', t2aa[p0:p1], ovvv)
            u2aa[:,p0:p1] += lib.einsum('ie,mbea->imab', t1a, ovvv.conj())
            ovvv = tmp1aa = None

    if nvirb > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            u1b += 0.5*lib.einsum('MIEF,MEAF->IA', t2bb[p0:p1], OVVV)
            u2bb[:,p0:p1] += lib.einsum('ie,mbea->imab', t1b, OVVV.conj())
            OVVV = tmp1bb = None

    if nvirb > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            u1b += lib.einsum('mIeF,meAF->IA', t2ab[p0:p1], ovVV)
            u2ab[p0:p1] += lib.einsum('IE,maEB->mIaB', t1b, ovVV.conj())
            ovVV = tmp1ab = None

    if nvira > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            u1a += lib.einsum('iMfE,MEaf->ia', t2ab[:,p0:p1], OVvv)
            u2ab[:,p0:p1] += lib.einsum('ie,MBea->iMaB', t1a, OVvv.conj())
            OVvv = tmp1abba = None

    eris_ovov = np.asarray(eris.ovov)
    eris_ovoo = np.asarray(eris.ovoo)
    Woooo = np.asarray(eris.oooo).transpose(0,2,1,3)
    u2aa += lib.einsum('mnab,mnij->ijab', tauaa, Woooo*.5)
    Woooo = tauaa = None
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    u1a += 0.5*lib.einsum('mnae,meni->ia', t2aa, ovoo)
    ovoo = eris_ovoo = None

    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    u2aa += ovov.conj().transpose(0,2,1,3) * .5
    eirs_ovov = ovov = tmpaa = tilaa = None

    eris_OVOV = np.asarray(eris.OVOV)
    eris_OVOO = np.asarray(eris.OVOO)
    WOOOO = np.asarray(eris.OOOO).transpose(0,2,1,3)
    u2bb += lib.einsum('mnab,mnij->ijab', taubb, WOOOO*.5)
    WOOOO = taubb = None
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    u1b += 0.5*lib.einsum('mnae,meni->ia', t2bb, OVOO)
    OVOO = eris_OVOO = None

    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    u2bb += OVOV.conj().transpose(0,2,1,3) * .5
    eris_OVOV = OVOV = tmpbb = tilbb = None

    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    u1a -= lib.einsum('nMaE,MEni->ia', t2ab, eris_OVoo)
    u1b -= lib.einsum('mNeA,meNI->IA', t2ab, eris_ovOO)
    WoOoO = np.asarray(eris.ooOO).transpose(0,2,1,3)
    eris_OVoo = eris_ovOO = None

    eris_ovOV = np.asarray(eris.ovOV)
    u2ab += lib.einsum('mNaB,mNiJ->iJaB', tauab, WoOoO)
    WoOoO = None

    u2ab += eris_ovOV.conj().transpose(0,2,1,3)
    tmpabab = tmpbaba = tilab = None

    Fova = fova
    Fovb = fovb
    u1a += fova.conj()
    u1a += np.einsum('ie,ae->ia', t1a, Fvva)
    u1a -= np.einsum('ma,mi->ia', t1a, Fooa)
    u1a -= np.einsum('imea,me->ia', t2aa, Fova)
    u1a += np.einsum('iMaE,ME->ia', t2ab, Fovb)
    u1b += fovb.conj()
    u1b += np.einsum('ie,ae->ia',t1b,Fvvb)
    u1b -= np.einsum('ma,mi->ia',t1b,Foob)
    u1b -= np.einsum('imea,me->ia', t2bb, Fovb)
    u1b += np.einsum('mIeA,me->IA', t2ab, Fova)

    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    wovvo -= eris_oovv.transpose(0,2,3,1)
    wovvo += eris_ovvo.transpose(0,2,1,3)
    oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
    u1a-= np.einsum('nf,niaf->ia', t1a,      oovv)
    eris_ovvo = eris_oovv = oovv = tmp1aa = None

    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    wOVVO -= eris_OOVV.transpose(0,2,3,1)
    wOVVO += eris_OVVO.transpose(0,2,1,3)
    OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
    u1b-= np.einsum('nf,niaf->ia', t1b,      OOVV)
    eris_OVVO = eris_OOVV = OOVV = None

    eris_ooVV = np.asarray(eris.ooVV)
    eris_ovVO = np.asarray(eris.ovVO)
    woVVo -= eris_ooVV.transpose(0,2,3,1)
    woVvO += eris_ovVO.transpose(0,2,1,3)
    u1b+= np.einsum('nf,nfAI->IA', t1a, eris_ovVO)
    eris_ooVV = eris_ovVo = tmp1ab = None

    eris_OOvv = np.asarray(eris.OOvv)
    eris_OVvo = np.asarray(eris.OVvo)
    wOvvO -= eris_OOvv.transpose(0,2,3,1)
    wOvVo += eris_OVvo.transpose(0,2,1,3)
    u1a+= np.einsum('NF,NFai->ia', t1b, eris_OVvo)
    eris_OOvv = eris_OVvO = tmp1ba = None

    u2aa += 2*lib.einsum('imae,mbej->ijab', t2aa, wovvo)
    u2aa += 2*lib.einsum('iMaE,MbEj->ijab', t2ab, wOvVo)
    u2bb += 2*lib.einsum('imae,mbej->ijab', t2bb, wOVVO)
    u2bb += 2*lib.einsum('mIeA,mBeJ->IJAB', t2ab, woVvO)
    u2ab += lib.einsum('imae,mBeJ->iJaB', t2aa, woVvO)
    u2ab += lib.einsum('iMaE,MBEJ->iJaB', t2ab, wOVVO)
    u2ab += lib.einsum('iMeA,MbeJ->iJbA', t2ab, wOvvO)
    u2ab += lib.einsum('IMAE,MbEj->jIbA', t2bb, wOvVo)
    u2ab += lib.einsum('mIeA,mbej->jIbA', t2ab, wovvo)
    u2ab += lib.einsum('mIaE,mBEj->jIaB', t2ab, woVVo)
    wovvo = wOVVO = woVvO = wOvVo = woVVo = wOvvO = None

    Ftmpa = Fvva.copy()
    Ftmpb = Fvvb.copy()
    u2aa += lib.einsum('ijae,be->ijab', t2aa, Ftmpa)
    u2bb += lib.einsum('ijae,be->ijab', t2bb, Ftmpb)
    u2ab += lib.einsum('iJaE,BE->iJaB', t2ab, Ftmpb)
    u2ab += lib.einsum('iJeA,be->iJbA', t2ab, Ftmpa)
    Ftmpa = Fooa.copy()
    Ftmpb = Foob.copy()
    u2aa -= lib.einsum('imab,mj->ijab', t2aa, Ftmpa)
    u2bb -= lib.einsum('imab,mj->ijab', t2bb, Ftmpb)
    u2ab -= lib.einsum('iMaB,MJ->iJaB', t2ab, Ftmpb)
    u2ab -= lib.einsum('mIaB,mj->jIaB', t2ab, Ftmpa)

    eris_ovoo = np.asarray(eris.ovoo).conj()
    eris_OVOO = np.asarray(eris.OVOO).conj()
    eris_OVoo = np.asarray(eris.OVoo).conj()
    eris_ovOO = np.asarray(eris.ovOO).conj()
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    u2aa -= lib.einsum('ma,jbim->ijab', t1a, ovoo)
    u2bb -= lib.einsum('ma,jbim->ijab', t1b, OVOO)
    u2ab -= lib.einsum('ma,JBim->iJaB', t1a, eris_OVoo)
    u2ab -= lib.einsum('MA,ibJM->iJbA', t1b, eris_ovOO)
    eris_ovoo = eris_OVoo = eris_OVOO = eris_ovOO = None

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)

    e = energy_ci(t1, t2, eris)
    u1a -= e * t1a
    u1b -= e * t1b
    u2aa -= e * t2aa
    u2ab -= e * t2ab
    u2bb -= e * t2bb

    time0 = log.timer_debug1('update t1 t2', *time0)
    t1new = -u1a, -u1b
    t2new = -u2aa, -u2ab, -u2bb
    return t1new, t2new

def update_ci_(cc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    Fooa = eris.focka[:nocca,:nocca]
    Foob = eris.fockb[:noccb,:noccb]
    Fvva = eris.focka[nocca:,nocca:]
    Fvvb = eris.fockb[noccb:,noccb:]

    u1a = fova.copy()
    u1b = fovb.copy()
    u1a += einsum('ic,ac->ia',t1a,Fvva)
    u1a -= einsum('ik,ka->ia',Fooa,t1a)
    u1a += einsum('kcai,kc->ia',eris.ovvo,t1a)
    u1a -= einsum('kiac,kc->ia',eris.oovv,t1a)
    u1a += einsum('KCai,KC->ia',eris.OVvo,t1b)
    u1a += einsum('kc,ikac->ia',fova,t2aa)
    u1a += einsum('KC,iKaC->ia',fovb,t2ab)
    u1a += 0.5*einsum('kcad,kicd->ia',eris.ovvv,t2aa)
    u1a -= 0.5*einsum('kdac,kicd->ia',eris.ovvv,t2aa)


    e = energy_ci(t1, t2, eris)
    u1a -= e * t1a
    u1b -= e * t1b
    u2aa -= e * t2aa
    u2ab -= e * t2ab
    u2bb -= e * t2bb

    time0 = log.timer_debug1('update t1 t2', *time0)
    t1new = -u1a, -u1b
    t2new = -u2aa, -u2ab, -u2bb
    return t1new, t2new

def xy2t(x1, y1, x2, y2):
    t1a = np.dot(y1[0],np.linalg.inv(x1[0])).T
    t1b = np.dot(y1[1],np.linalg.inv(x1[1])).T
    noa, nva, nob, nvb = x2[1].shape
    na, nb = noa*nva, nob*nvb
    xaa = x2[0].reshape((na,na))
    xab = x2[1].reshape((na,nb))
    xba = x2[2].reshape((nb,na))
    xbb = x2[3].reshape((nb,nb))
    yaa = y2[0].reshape((na,na))
    yab = y2[1].reshape((na,nb))
    yba = y2[2].reshape((nb,na))
    ybb = y2[3].reshape((nb,nb))
    x = np.block([[xaa,xab],[xba,xbb]])
    y = np.block([[yaa,yab],[yba,ybb]])
    t = np.dot(y, np.linalg.inv(x))
    t2aa = t[:na,:na].reshape((noa,nva,noa,nva)).transpose(0,2,1,3)
    t2ab = t[:na,na:].reshape((noa,nva,nob,nvb)).transpose(0,2,1,3)
    t2bb = t[na:,na:].reshape((nob,nvb,nob,nvb)).transpose(0,2,1,3)
    return (t1a, t1b), (t2aa, t2ab, t2bb)

def t2xy(t1, t2):
    noa, nob, nva, nvb = t2[1].shape
    x1 = np.eye(noa), np.eye(nob)
    y1 = t1[0].T, t1[1].T
    yaa = t2[0].transpose(0,2,1,3) 
    yab = t2[1].transpose(0,2,1,3) 
    yba = t2[1].transpose(1,3,0,2) 
    ybb = t2[2].transpose(0,2,1,3) 
    xab, xba = np.zeros_like(yab), np.zeros_like(yba)
    Iva, Ivb = np.eye(nva), np.eye(nvb)
    xaa = einsum('ij,ab->iajb',x1[0], Iva)
    xbb = einsum('ij,ab->iajb',x1[1], Ivb)
    y2 = yaa, yab, yba, ybb
    x2 = xaa, xab, xba, xbb
    return x1, y1, x2, y2

def update_ccs(cc, x, y, eris):
    nva, noa = y[0].shape
    nvb, nob = y[1].shape
    fa, fb = eris.focka, eris.fockb
    dxa = np.dot(fa[:noa,:noa], x[0]) + np.dot(fa[:noa,noa:], y[0])
    dya = np.dot(fa[noa:,:noa], x[0]) + np.dot(fa[noa:,noa:], y[0])
    dxb = np.dot(fb[:nob,:nob], x[1]) + np.dot(fb[:nob,nob:], y[1]) 
    dyb = np.dot(fb[nob:,:nob], x[1]) + np.dot(fb[nob:,nob:], y[1])
    return (-dxa, -dxb), (-dya, -dyb) 

def update_rpa(cc, x, y, eris):
    def contract(eia, ovvo, oovv, ovVO, ovov, ovOV, 
                 x1, x2, y1, y2):
        dx = np.multiply(eia, x1)
        dx += einsum('iabj,jbkc->iakc', ovvo, x1)
        dx -= einsum('ijba,jbkc->iakc', oovv, x1)
        dx += einsum('iaBJ,JBkc->iakc', ovVO, x2)
        dx += einsum('iajb,jbkc->iakc', ovov, y1)
        dx -= einsum('ibja,jbkc->iakc', ovov, y1)
        dx += einsum('iaJB,JBkc->iakc', ovOV, y2)
        return dx
    noa, nva, nob, nvb = x[1].shape
    eoa, eva = eris.focka.diagonal()[:noa], eris.focka.diagonal()[noa:]
    eob, evb = eris.focka.diagonal()[:nob], eris.focka.diagonal()[nob:]
    eia_a = eva[None,:,None,None] - eoa[:,None,None,None] 
    eia_b = evb[None,:,None,None] - eob[:,None,None,None]

    ovvo = eris.ovvo
    oovv = eris.oovv
    ovVO = eris.ovVO
    ovov = eris.ovov
    ovOV = eris.ovOV
    dxaa = contract(eia_a,ovvo,oovv,ovVO,ovov,ovOV,x[0],x[2],y[0],y[2])
    dxab = contract(eia_a,ovvo,oovv,ovVO,ovov,ovOV,x[1],x[3],y[1],y[3])
    dyaa = - contract(eia_a,ovvo,oovv,ovVO,ovov,ovOV,y[0],y[2],x[0],x[2])
    dyab = - contract(eia_a,ovvo,oovv,ovVO,ovov,ovOV,y[1],y[3],x[1],x[3])
    ovvo = eris.OVVO
    oovv = eris.OOVV
    ovVO = eris.OVvo
    ovov = eris.OVOV
    ovOV = eris.ovOV.transpose(2,3,0,1)
    dxbb = contract(eia_b,ovvo,oovv,ovVO,ovov,ovOV,x[3],x[1],y[3],y[1])
    dxba = contract(eia_b,ovvo,oovv,ovVO,ovov,ovOV,x[2],x[0],y[2],y[0])
    dybb = - contract(eia_b,ovvo,oovv,ovVO,ovov,ovOV,y[3],y[1],x[3],x[1])
    dyba = - contract(eia_b,ovvo,oovv,ovVO,ovov,ovOV,y[2],y[0],x[2],x[0])
    return (dxaa, dxab, dxba, dxbb), (dyaa, dyab, dyba, dybb)

def _ccs(mf, frozen):
    cc = uccsd.UCCSD(mf, frozen)
    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    
    eris = cc.ao2mo(cc.mo_coeff)
    wa, va = np.linalg.eigh(eris.focka)
    wb, vb = np.linalg.eigh(eris.fockb)
    print('diagonalized coeff: \n{}\n{}'.format(va[:,:noa],vb[:,:noa]))
    xa, ya, xb, yb = va[:noa,:noa], va[noa:,:noa], vb[:nob,:nob], vb[nob:,:nob]

    thresh = 0.001
    if abs(np.linalg.det(xa)) < thresh or abs(np.linalg.det(xb)) < thresh:
        return None, True
    else:
        t1a, t1b = np.dot(ya, np.linalg.inv(xa)).T, np.dot(yb, np.linalg.inv(xb)).T
        t2 = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
             np.zeros((nob,nob,nvb,nvb))
        eccs = cc.energy((t1a, t1b), t2, eris) 
        return eccs, False

def _rpa(mf, frozen=0):
    cc = uccsd.UCCSD(mf, frozen)
    eris = cc.ao2mo(cc.mo_coeff)
    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    na, nb = noa*nva, nob*nvb

    t1 = np.zeros((noa,nva)), np.zeros((nob,nvb))
    t2 = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,nob,nvb,nvb))
    _, _, x, y = t2xy(t1, t2)

    tl, bl = update_rpa(cc, x, y, eris)
    tr, br = update_rpa(cc, y, x, eris)
    tlaa, tlab, tlba, tlbb = tl
    blaa, blab, blba, blbb = bl
    traa, trab, trba, trbb = tr
    braa, brab, brba, brbb = br
    tlaa = tlaa.reshape((na,na))
    tlab = tlab.reshape((na,nb))
    tlba = tlba.reshape((nb,na))
    tlbb = tlbb.reshape((nb,nb))
    blaa = blaa.reshape((na,na))
    blab = blab.reshape((na,nb))
    blba = blba.reshape((nb,na))
    blbb = blbb.reshape((nb,nb))
    traa = traa.reshape((na,na))
    trab = trab.reshape((na,nb))
    trba = trba.reshape((nb,na))
    trbb = trbb.reshape((nb,nb))
    braa = braa.reshape((na,na))
    brab = brab.reshape((na,nb))
    brba = brba.reshape((nb,na))
    brbb = brbb.reshape((nb,nb))
    
    tl = np.block([[tlaa,tlab],[tlba,tlbb]])
    bl = np.block([[blaa,blab],[blba,blbb]])
    tr = np.block([[traa,trab],[trba,trbb]])
    br = np.block([[braa,brab],[brba,brbb]])
    M = np.block([[tl,tr],[bl,br]])
    w, v = scipy.linalg.eig(M)

    idx = np.argsort(w.real) 
    w = w[idx]
    v = v[:,idx]
    w_minus, w_plus = np.hsplit(w, 2)
    v_minus, v_plus = np.hsplit(v, 2)
    print('rpa coeff:\n{}'.format(np.fliplr(v_plus.real)))
#    v_minus = np.fliplr(v_minus)
#    v_minus = np.vstack((v_minus[na+nb:,:],v_minus[:na+nb,:]))
    print('check w split: {}'.format(np.linalg.norm(np.flip(w_minus)+w_plus)/w_minus.size))
#    print('w_minus: \n{}\n{}'.format(w_minus.real, w_minus.imag))
#    print('check v split: {}'.format(np.linalg.norm(v_plus-v_minus)/v_minus.size))
#    print(v_minus)

    x, y = np.vsplit(v_plus, 2)
    t = np.dot(y, np.linalg.inv(x))
    t2aa = t[:na,:na].reshape((noa,nva,noa,nva)).transpose(0,2,1,3)
    t2ab = t[:na,na:].reshape((noa,nva,nob,nvb)).transpose(0,2,1,3)
    t2bb = t[na:,na:].reshape((nob,nvb,nob,nvb)).transpose(0,2,1,3)
    erccd = cc.energy(t1, (t2aa, t2ab, t2bb), eris) 
    return w_plus, erccd
    

def kernel(cc, beta, dt, default='ccsd', lin='cisd'):
    tol = cc.conv_tol
    tolnormt = cc.conv_tol_normt

    assert(cc.mo_coeff is not None)
    assert(cc.mo_occ is not None)
    if cc.verbose >= logger.WARN:
        ccsd.CCSD.check_sanity(cc)
    ccsd.CCSD.dump_flags(cc)
    log = logger.new_logger(cc, cc.verbose)

    eris = cc.ao2mo(cc.mo_coeff)

    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    t1 = np.zeros((noa, nva)), np.zeros((nob, nvb))
    t2 = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,nob,nvb,nvb))
    c1 = c2 = None
    x1 = x2 = y1 = y2 = None
    method = default

    cput1 = cput0 = (time.clock(), time.time())
    eold = 0
    conv = False
    eccsd = cc.energy(t1, t2, eris)
    log.info('Init E(CCSD) = %.15g', eccsd)

    steps = np.arange(0.0, beta, dt)
    for istep in steps:
        if method == 'ccsd':
            dt1, dt2 = update_cc(cc, t1, t2, eris)
            t1new = t1[0]+dt1[0]*dt, t1[1]+dt1[1]*dt
            t2new = t2[0]+dt2[0]*dt, t2[1]+dt2[1]*dt, t2[2]+dt2[2]*dt
        if method == 'cisd':
            if c1 is None:
                c1, c2 = t2ci(t1, t2)
            dc1, dc2 = update_ci(cc, c1, c2, eris)
            c1 = c1[0]+dc1[0]*dt, c1[1]+dc1[1]*dt
            c2 = c2[0]+dc2[0]*dt, c2[1]+dc2[1]*dt, c2[2]+dc2[2]*dt
            t1new, t2new = ci2t(c1, c2)
        if method == 'rpa':
            if x1 is None:
                x1, y1, x2, y2 = t2xy(t1, t2)
            dx1, dy1 = update_ccs(cc, x1, y1, eris)
            x1 = x1[0]+dx1[0]*dt, x1[1]+dx1[1]*dt
            y1 = y1[0]+dy1[0]*dt, y1[1]+dy1[1]*dt
            dx2, dy2 = update_rpa(cc, x2, y2, eris)
            x2 = x2[0]+dx2[0]*dt,x2[1]+dx2[1]*dt,x2[2]+dx2[2]*dt,x2[3]+dx2[3]*dt
            y2 = y2[0]+dy2[0]*dt,y2[1]+dy2[1]*dt,y2[2]+dy2[2]*dt,y2[3]+dy2[3]*dt
            t1new, t2new = xy2t(x1, y1, x2, y2)

        normt = np.linalg.norm(cc.amplitudes_to_vector(t1new, t2new) -
                               cc.amplitudes_to_vector(t1, t2))
        t1, t2 = t1new, t2new
        t1new = t2new = None
        eold, eccsd = eccsd, cc.energy(t1, t2, eris)

        log.info('tau = %.4g  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)

        if abs(eccsd) > 100.0:
            method = lin 
#            break
        else: 
            x1 = y1 = x2 = y2 = None
            method = default
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            beta = istep
            break
    log.timer('CCSD', *cput0)

#    ca, cb = np.vstack((x1[0],y1[0])), np.vstack((x1[1],y1[1]))
#    for i in range(ca.shape[1]):
#        ca[:,i] /= np.linalg.norm(ca[:,i])
#    for i in range(cb.shape[1]):
#        cb[:,i] /= np.linalg.norm(cb[:,i])
#    print('it coeff:\n{}\n{}'.format(ca, cb))

#    na, nb = noa*nva, nob*nvb
#    xaa, xab, xba, xbb = x2
#    yaa, yab, yba, ybb = y2
#    xaa = xaa.reshape((na,na))
#    xab = xab.reshape((na,nb))
#    xba = xba.reshape((nb,na))
#    xbb = xbb.reshape((nb,nb))
#    yaa = yaa.reshape((na,na))
#    yab = yab.reshape((na,nb))
#    yba = yba.reshape((nb,na))
#    ybb = ybb.reshape((nb,nb))
#    c = np.block([[xaa,xab],[xba,xbb],[yaa,yab],[yba,ybb]])
#    for i in range(c.shape[1]):
#        c[:,i] /= np.linalg.norm(c[:,i])
#    print('it coeff:\n{}'.format(c))

    cc.converged = conv
    cc.e_corr = eccsd
    cc.t1, cc.t2 = t1, t2
    ccsd.CCSD._finalize(cc)
    return beta


class itCCSD(ccsd.CCSD):

    conv_tol = getattr(__config__, 'cc_uccsd_UCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_uccsd_UCCSD_conv_tol_normt', 1e-6)

# Attribute frozen can be
# * An integer : The same number of inner-most alpha and beta orbitals are frozen
# * One list : Same alpha and beta orbital indices to be frozen
# * A pair of list : First list is the orbital indices to be frozen for alpha
#       orbitals, second list is for beta orbitals
    def __init__(self, mf, default='ccsd', lin='cisd',
                 ncas=None, nelecas=None, 
                 occslst=None, 
                 ci=None, ndet=None, cutoff=None, 
                 frozen=0, beta=10.0, dt=0.1):

        nea, neb = mf.mol.nelec
        moa, mob = mf.mo_coeff
        self.mo0 = moa.copy(), mob.copy()
        nmoa, nmob = moa.shape[1], mob.shape[1]
        self.nfc, self.nfv = get_frozen(mf, frozen)
        print('nmo: {}, nele: {}'.format((nmoa, nmob), (nea, neb)))
        print('frozen core: {}, virtual: {}'.format(self.nfc, self.nfv))

        self.occ = get_occ(mf, ncas, nelecas, occslst, ci, ndet, cutoff)
        ndet = self.occ.shape[0]
        self.vir = np.empty((ndet,nmoa+nmob-nea-neb),dtype=int)
        self.beta = np.ones(ndet) * beta
        self.dt = dt

        noa, nob = nea - self.nfc[0], neb - self.nfc[1]
        nva, nvb = nmoa - nea - self.nfv[0], nmob - neb - self.nfv[1] 
        self.t1a = np.zeros((ndet, noa, nva))
        self.t1b = np.zeros((ndet, nob, nvb))
        self.t2aa = np.zeros((ndet, noa, noa, nva, nva))
        self.t2ab = np.zeros((ndet, noa, nob, nva, nvb))
        self.t2bb = np.zeros((ndet, nob, nob, nvb, nvb))

        self.cc = uccsd.UCCSD(mf, frozen)
        self.default = default
        self.lin = lin

    def ccsd(self):
        ndet = self.occ.shape[0]
        eccsd = np.zeros(ndet)
        div = []
        for I in range(ndet):
            print('\nreference {} occupation: {}'.format(I, self.occ[I,:]))
            mo_coeff, self.occ[I,:], self.vir[I,:] = \
                perm_mo(self.cc._scf, self.mo0, self.occ[I,:])
            self.cc.mo_coeff = self.cc._scf.mo_coeff = mo_coeff
            e_ref = ref_energy(self.cc._scf)
            self.cc._scf.e_tot = e_ref + self.cc._scf.mol.energy_nuc()
            print('reference energy: {}'.format(self.cc._scf.e_tot))
#            _, erccd = _rpa(self.cc._scf, self.cc.frozen)
#            print('rCCD energy: {}'.format(erccd))
#            eccs, ortho = _ccs(self.cc._scf, self.cc.frozen)
#            if ortho:
#                continue
#            print('CCS energy: {}'.format(eccs))
            self.beta[I] = kernel(self.cc, self.beta[I], self.dt,
                                  default=self.default, lin=self.lin)
            if abs(self.cc.e_corr) > 10.0:
                print('Divergent energy!')
                div.append(I)
            else:
                t1a, t1b = self.cc.t1
                t2aa, t2ab, t2bb = self.cc.t2
                self.t1a[I,:,:] = t1a.copy()
                self.t1b[I,:,:] = t1b.copy()
                self.t2aa[I,:,:,:,:] = t2aa.copy()
                self.t2ab[I,:,:,:,:] = t2ab.copy()
                self.t2bb[I,:,:,:,:] = t2bb.copy()
                eccsd[I] = e_ref + self.cc.e_corr
                print('max element: t1a, t1b\n{}, {}'.format(
                      np.amax(abs(t1a)),np.amax(abs(t1b))))
                print('max element: t2aa, t2ab, t2bb\n{}, {}, {}'.format(
                      np.amax(abs(t2aa)),np.amax(abs(t2ab)),np.amax(abs(t2bb))))
        return eccsd, div

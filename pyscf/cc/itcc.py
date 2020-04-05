import time, math, scipy
import numpy as np

from pyscf import lib, ao2mo, __config__
from pyscf.lib import logger
from pyscf.cc import ccsd, uccsd
from pyscf.ci import ucisd
from pyscf.fci.cistring import _gen_occslst

np.set_printoptions(precision=8,suppress=True)
einsum = lib.einsum
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

make_tau = uccsd.make_tau
make_tau_aa = uccsd.make_tau_aa
make_tau_ab = uccsd.make_tau_ab

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
    moa, mob = np.hstack((moa_occ,moa_vir)), np.hstack((mob_occ,mob_vir))
    return (moa, mob)

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
    u1, u2 = cc.update_amps(t1, t2, eris)
    u1a, u1b = u1
    u2aa, u2ab, u2bb = u2
    noa, nob, nva, nvb = t2ab.shape
    mo_ea_o = eris.mo_energy[0][:noa]
    mo_ea_v = eris.mo_energy[0][noa:] + cc.level_shift
    mo_eb_o = eris.mo_energy[1][:nob]
    mo_eb_v = eris.mo_energy[1][nob:] + cc.level_shift
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    eijab_aa = lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    eijab_ab = lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    eijab_bb = lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
   
    u1a *= eia_a
    u1b *= eia_b
    u2aa *= eijab_aa
    u2ab *= eijab_ab
    u2bb *= eijab_bb

    u1a -= eia_a * t1[0] 
    u1b -= eia_b * t1[1] 
    u2aa -= eijab_aa * t2[0]
    u2ab -= eijab_ab * t2[1]
    u2bb -= eijab_bb * t2[2]
    return (-u1a, -u1b), (-u2aa, -u2ab, -u2bb)

def t2ci(t1, t2):
    t1a, t1b = t1
    c2aa, c2ab, c2bb = t2
    c2aa += einsum('ia,jb->ijab',t1a,t1a) - einsum('ib,ja->ijab',t1a,t1a) 
    c2bb += einsum('ia,jb->ijab',t1b,t1b) - einsum('ib,ja->ijab',t1b,t1b)
    c2ab += einsum('ia,jb->ijab',t1a,t1b)
    civec = ucisd.amplitudes_to_cisdvec(1.0, t1, (c2aa, c2ab, c2bb))
    return civec/np.linalg.norm(civec)

def ci2t(cc, civec):
    nocc = cc.get_nocc()
    nmo = cc.get_nmo()
    c0, c1, c2 = ucisd.cisdvec_to_amplitudes(civec, nmo, nocc)
    c1 = c1[0]/c0, c1[1]/c0
    c2 = c2[0]/c0, c2[1]/c0, c2[2]/c0

    t1a, t1b = c1
    t2aa, t2ab, t2bb = c2

    t2aa -= einsum('ia,jb->ijab',t1a,t1a) - einsum('ib,ja->ijab',t1a,t1a) 
    t2bb -= einsum('ia,jb->ijab',t1b,t1b) - einsum('ib,ja->ijab',t1b,t1b)
    t2ab -= einsum('ia,jb->ijab',t1a,t1b)
    return (t1a, t1b), (t2aa, t2ab, t2bb)  

def update_ci(cc, civec, eris):
    ci = ucisd.UCISD(cc._scf, frozen=cc.frozen)
    return -ci.contract(civec, eris)

def xy2t_1(x1, y1):
    t1a = np.dot(y1[0],np.linalg.inv(x1[0])).T
    t1b = np.dot(y1[1],np.linalg.inv(x1[1])).T
    return (t1a, t1b)

def xy2t_2(x2, y2):
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
    return (t2aa, t2ab, t2bb)

def t2xy_1(t1):
    noa, nva = t1[0].shape
    nob, nvb = t1[1].shape
    x1 = np.eye(noa), np.eye(nob)
    y1 = t1[0].T, t1[1].T
    return x1, y1

def t2xy_2(t2):
    noa, nob, nva, nvb = t2[1].shape
    yaa = t2[0].transpose(0,2,1,3) 
    yab = t2[1].transpose(0,2,1,3) 
    yba = t2[1].transpose(1,3,0,2) 
    ybb = t2[2].transpose(0,2,1,3) 
    xab, xba = np.zeros_like(yab), np.zeros_like(yba)
    Ioa, Iob = np.eye(noa), np.eye(nob)
    Iva, Ivb = np.eye(nva), np.eye(nvb)
    xaa = einsum('ij,ab->iajb',Ioa, Iva)
    xbb = einsum('ij,ab->iajb',Iob, Ivb)
    y2 = yaa, yab, yba, ybb
    x2 = xaa, xab, xba, xbb
    return x2, y2

def update_ccs(x, y, eris):
    nva, noa = y[0].shape
    nvb, nob = y[1].shape
    fa, fb = eris.focka, eris.fockb
    dxa = np.dot(fa[:noa,:noa], x[0]) + np.dot(fa[:noa,noa:], y[0])
    dya = np.dot(fa[noa:,:noa], x[0]) + np.dot(fa[noa:,noa:], y[0])
    dxb = np.dot(fb[:nob,:nob], x[1]) + np.dot(fb[:nob,nob:], y[1]) 
    dyb = np.dot(fb[nob:,:nob], x[1]) + np.dot(fb[nob:,nob:], y[1])
    return (-dxa, -dxb), (-dya, -dyb) 

def update_rpa(x, y, eris):
    def contract(eia, ovvo, oovv, ovVO, ovov, ovOV, 
                 x1, x2, y1, y2):
        dx = eia * x1
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

def orthogonalize(x, y):
    xaa, xab, xba, xbb = x
    yaa, yab, yba, ybb = y
    noa, nva, nob, nvb = x[1].shape 
    na, nb = noa*nva, nob*nvb
    uxaa = x[0].reshape((noa,nva,na))
    uxab = x[1].reshape((noa,nva,nb))
    uxba = x[2].reshape((nob,nvb,na))
    uxbb = x[3].reshape((nob,nvb,nb))
    uyaa = y[0].reshape((noa,nva,na))
    uyab = y[1].reshape((noa,nva,nb))
    uyba = y[2].reshape((nob,nvb,na))
    uybb = y[3].reshape((nob,nvb,nb))
    vxaa = uxaa.copy() 
    vxab = uxab.copy()
    vxba = uxba.copy()
    vxbb = uxbb.copy()
    vyaa = uyaa.copy()
    vyab = uyab.copy()
    vyba = uyba.copy()
    vybb = uybb.copy()
    for k in range(1,na):
        for j in range(k):
            num  = einsum('ia,ia',vxaa[:,:,k],uxaa[:,:,j])
            num += einsum('IA,IA',vxba[:,:,k],uxba[:,:,j])
            num -= einsum('ia,ia',vyaa[:,:,k],uyaa[:,:,j])
            num -= einsum('IA,IA',vyba[:,:,k],uyba[:,:,j])
            denom  = einsum('ia,ia',uxaa[:,:,j],uxaa[:,:,j])
            denom += einsum('IA,IA',uxba[:,:,j],uxba[:,:,j])
            denom -= einsum('ia,ia',uyaa[:,:,j],uyaa[:,:,j])
            denom -= einsum('IA,IA',uyba[:,:,j],uyba[:,:,j])  
            uxaa[:,:,k] -= uxaa[:,:,j]*num/denom
            uxba[:,:,k] -= uxba[:,:,j]*num/denom
            uyaa[:,:,k] -= uyaa[:,:,j]*num/denom
            uyba[:,:,k] -= uyba[:,:,j]*num/denom
    for k in range(nb):
        for j in range(na):
            num  = einsum('ia,ia',vxab[:,:,k],uxaa[:,:,j])
            num += einsum('IA,IA',vxbb[:,:,k],uxba[:,:,j])
            num -= einsum('ia,ia',vyab[:,:,k],uyaa[:,:,j])
            num -= einsum('IA,IA',vybb[:,:,k],uyba[:,:,j])
            denom  = einsum('ia,ia',uxaa[:,:,j],uxaa[:,:,j])
            denom += einsum('IA,IA',uxba[:,:,j],uxba[:,:,j])
            denom -= einsum('ia,ia',uyaa[:,:,j],uyaa[:,:,j])
            denom -= einsum('IA,IA',uyba[:,:,j],uyba[:,:,j])  
            uxab[:,:,k] -= uxaa[:,:,j]*num/denom
            uxbb[:,:,k] -= uxba[:,:,j]*num/denom
            uyab[:,:,k] -= uyaa[:,:,j]*num/denom
            uybb[:,:,k] -= uyba[:,:,j]*num/denom
        for j in range(k):
            num  = einsum('ia,ia',vxab[:,:,k],uxab[:,:,j])
            num += einsum('IA,IA',vxbb[:,:,k],uxbb[:,:,j])
            num -= einsum('ia,ia',vyab[:,:,k],uyab[:,:,j])
            num -= einsum('IA,IA',vybb[:,:,k],uybb[:,:,j])
            denom  = einsum('ia,ia',uxab[:,:,j],uxab[:,:,j])
            denom += einsum('IA,IA',uxbb[:,:,j],uxbb[:,:,j])
            denom -= einsum('ia,ia',uyab[:,:,j],uyab[:,:,j])
            denom -= einsum('IA,IA',uybb[:,:,j],uybb[:,:,j])  
            uxab[:,:,k] -= uxab[:,:,j]*num/denom
            uxbb[:,:,k] -= uxbb[:,:,j]*num/denom
            uyab[:,:,k] -= uyab[:,:,j]*num/denom
            uybb[:,:,k] -= uybb[:,:,j]*num/denom
    uxaa = uxaa.reshape((noa,nva,noa,nva))
    uxab = uxab.reshape((noa,nva,nob,nvb))
    uxba = uxba.reshape((nob,nvb,noa,nva))
    uxbb = uxbb.reshape((nob,nvb,nob,nvb))
    uyaa = uyaa.reshape((noa,nva,noa,nva))
    uyab = uyab.reshape((noa,nva,nob,nvb))
    uyba = uyba.reshape((nob,nvb,noa,nva))
    uybb = uybb.reshape((nob,nvb,nob,nvb))
    return (uxaa,uxab,uxba,uxbb), (uyaa,uyab,uyba,uybb)

def update_rccd(t2, eris):
    t2aa, t2ab, t2bb = t2
    noa, nob, nva,nvb = t2ab.shape
    oovv = eris.ovov.transpose(0,2,1,3) - eris.ovov.transpose(0,2,3,1)
    OOVV = eris.OVOV.transpose(0,2,1,3) - eris.OVOV.transpose(0,2,3,1)
    oOvV = eris.ovOV.transpose(0,2,1,3)

    u2aa  = einsum('iack,jkbc->ijab',eris.ovvo,t2aa)
    u2aa -= einsum('ikca,jkbc->ijab',eris.oovv,t2aa)
    u2aa += einsum('iaCK,jKbC->ijab',eris.ovVO,t2ab)
    u2aa += u2aa.transpose(1,0,3,2)
    u2aa += oovv.copy()
    temp  = einsum('ikac,klcd->ilad',t2aa,oovv)
    temp += einsum('iKaC,lKdC->ilad',t2ab,oOvV)
    u2aa += einsum('ilad,jlbd->ijab',temp,t2aa)
    temp  = einsum('iKaC,KLCD->iLaD',t2ab,OOVV)
    temp += einsum('ikac,kLcD->iLaD',t2aa,oOvV)
    u2aa += einsum('iLaD,jLbD->ijab',temp,t2ab)

    u2bb  = einsum('IACK,JKBC->IJAB',eris.OVVO,t2bb)
    u2bb -= einsum('IKCA,JKBC->IJAB',eris.OOVV,t2bb)
    u2bb += einsum('IAck,kJcB->IJAB',eris.OVvo,t2ab)
    u2bb += u2bb.transpose(1,0,3,2)
    u2bb += OOVV.copy()
    temp  = einsum('IKAC,KLCD->ILAD',t2bb,OOVV)
    temp += einsum('kIcA,kLcD->ILAD',t2ab,oOvV)
    u2bb += einsum('ILAD,JLBD->IJAB',temp,t2bb)
    temp  = einsum('kIcA,klcd->IlAd',t2ab,oovv)
    temp += einsum('IKAC,lKdC->IlAd',t2bb,oOvV)
    u2bb += einsum('IlAd,lJdB->IJAB',temp,t2ab)

    u2ab  = oOvV.copy()
    u2ab += einsum('iack,kJcB->iJaB',eris.ovvo,t2ab)
    u2ab -= einsum('ikca,kJcB->iJaB',eris.oovv,t2ab)
    u2ab += einsum('iaCK,JKBC->iJaB',eris.ovVO,t2bb)
    u2ab += einsum('JBCK,iKaC->iJaB',eris.OVVO,t2ab)
    u2ab -= einsum('JKCB,iKaC->iJaB',eris.OOVV,t2ab)
    u2ab += einsum('JBck,ikac->iJaB',eris.OVvo,t2aa)
    temp  = einsum('ikac,klcd->ilad',t2aa,oovv)
    temp += einsum('iKaC,lKdC->ilad',t2ab,oOvV)
    u2ab += einsum('ilad,lJdB->iJaB',temp,t2ab)
    temp  = einsum('ikac,kLcD->iLaD',t2aa,oOvV)
    temp += einsum('iKaC,KLCD->iLaD',t2ab,OOVV)
    u2ab += einsum('iLaD,JLBD->iJaB',temp,t2bb)

    mo_ea_o = eris.mo_energy[0][:noa]
    mo_ea_v = eris.mo_energy[0][noa:]
    mo_eb_o = eris.mo_energy[1][:nob]
    mo_eb_v = eris.mo_energy[1][nob:]
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    eijab_aa = lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    eijab_ab = lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    eijab_bb = lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
    u2aa -= eijab_aa * t2aa
    u2ab -= eijab_ab * t2ab
    u2bb -= eijab_bb * t2bb
    return (-u2aa, -u2ab, -u2bb)

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
    print(eris.focka)
    print(eris.fockb)

    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    t1 = np.zeros((noa, nva)), np.zeros((nob, nvb))
    t2 = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,nob,nvb,nvb))
    civec = None
    x1 = x2 = y1 = y2 = None
    method = default

    cput1 = cput0 = (time.clock(), time.time())
    eold = 0
    conv = False
    eccsd = cc.energy(t1, t2, eris)
    log.info('Init E(CCSD) = %.15g', eccsd)

    steps = np.arange(0.0, beta, dt)
    for istep in steps:
#        if method == 'ccsd':
#            dt1, dt2 = update_cc(cc, t1, t2, eris)
#            t1new = t1[0]+dt1[0]*dt, t1[1]+dt1[1]*dt
#            t2new = t2[0]+dt2[0]*dt, t2[1]+dt2[1]*dt, t2[2]+dt2[2]*dt
#        if method == 'cisd':
#            if civec is None:
#                civec = t2ci(t1, t2)
#            dc = update_ci(cc, civec, eris)
#            civec = civec + dc*dt
#            t1new, t2new = ci2t(cc, civec)
        if method == 'rpa':
#            if x1 is None:
#                x1, y1 = t2xy_1(t1)
#            dx1, dy1 = update_ccs(x1, y1, eris)
#            x1 = x1[0]+dx1[0]*dt, x1[1]+dx1[1]*dt
#            y1 = y1[0]+dy1[0]*dt, y1[1]+dy1[1]*dt
#            t1new = xy2t_1(x1, y1)
            if x2 is None:
                x2, y2 = t2xy_2(t2)
            dx2, dy2 = update_rpa(x2, y2, eris)
            x2 = x2[0]+dx2[0]*dt,x2[1]+dx2[1]*dt,x2[2]+dx2[2]*dt,x2[3]+dx2[3]*dt
            y2 = y2[0]+dy2[0]*dt,y2[1]+dy2[1]*dt,y2[2]+dy2[2]*dt,y2[3]+dy2[3]*dt
            x2, y2 = orthogonalize(x2, y2)
            t2new = xy2t_2(x2, y2)
        if method == 'rccd':
            dt2 = update_rccd(t2, eris)
            t2new = t2[0]+dt2[0]*dt, t2[1]+dt2[1]*dt, t2[2]+dt2[2]*dt

        t2new = t2
        normt = np.linalg.norm(cc.amplitudes_to_vector(t1new, t2new) -
                               cc.amplitudes_to_vector(t1, t2))
        t1, t2 = t1new, t2new
        t1new = t2new = None
        eold, eccsd = eccsd, cc.energy(t1, t2, eris)

        log.info('tau = %.4g  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)

#        if abs(eccsd) > 10.0:
#            method = lin 
#            break
#        else: 
#            x1 = y1 = None
#            x2 = y2 = None
#            method = default
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            beta = istep
            break
    log.timer('CCSD', *cput0)

    ca, cb = np.vstack((x1[0],y1[0])), np.vstack((x1[1],y1[1]))
    for i in range(ca.shape[1]):
        ca[:,i] /= np.linalg.norm(ca[:,i])
    for i in range(cb.shape[1]):
        cb[:,i] /= np.linalg.norm(cb[:,i])
    print('it coeff:\n{}\n{}'.format(ca, cb))

#    na, nb = noa*nva, nob*nvb
#    if default == 'rccd':
#        x2, y2 = t2xy_2(t2)
#    x2, y2 = orthogonalize(x2, y2)
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
#    x = np.block([[xaa,xab],[xba,xbb]])
#    y = np.block([[yaa,yab],[yba,ybb]])
#    print('T from imaginary time:\n{}'.format(np.dot(y, np.linalg.inv(x)).T))

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

        self.cc = uccsd.UCCSD(mf, frozen)
        nea, neb = mf.mol.nelec
        moa, mob = mf.mo_coeff
        self.mo0 = moa.copy(), mob.copy()
        nmoa, nmob = moa.shape[1], mob.shape[1]

        self.occ = get_occ(mf, ncas, nelecas, occslst, ci, ndet, cutoff)
        ndet = self.occ.shape[0]
        self.beta = np.ones(ndet) * beta
        self.dt = dt

        noa, nob = self.cc.get_nocc()
        nmoa, nmob = self.cc.get_nmo()
        nva, nvb = nmoa - noa, nmob - nob
        self.t1a = np.zeros((ndet, noa, nva))
        self.t1b = np.zeros((ndet, nob, nvb))
        self.t2aa = np.zeros((ndet, noa, noa, nva, nva))
        self.t2ab = np.zeros((ndet, noa, nob, nva, nvb))
        self.t2bb = np.zeros((ndet, nob, nob, nvb, nvb))

        self.default = default
        self.lin = lin

    def ccsd(self):
        ndet = self.occ.shape[0]
        eccsd = np.zeros(ndet)
        div = []
        for I in range(ndet):
            print('\nreference {} occupation: {}'.format(I, self.occ[I,:]))
            mo_coeff = perm_mo(self.cc._scf, self.mo0, self.occ[I,:])
            self.cc.mo_coeff = self.cc._scf.mo_coeff = mo_coeff
            e_ref = ref_energy(self.cc._scf)
            self.cc._scf.e_tot = e_ref + self.cc._scf.mol.energy_nuc()
#            ci = ucisd.UCISD(self.cc._scf, frozen=self.cc.frozen)
#            ci.kernel()
            print('reference energy: {}'.format(self.cc._scf.e_tot))
#            _rpa(self.cc._scf, self.cc.frozen)
            _ccs(self.cc._scf, self.cc.frozen)
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

def _ccs(mf, frozen):
    cc = uccsd.UCCSD(mf, frozen)
    eris = cc.ao2mo(cc.mo_coeff)
    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    
    wa, va = np.linalg.eigh(eris.focka)
    wb, vb = np.linalg.eigh(eris.fockb)
    print('diagonalized coeff: \n{}\n{}'.format(va[:,:noa],vb[:,:noa]))
    return

def _rpa(mf, frozen=0):
    cc = uccsd.UCCSD(mf, frozen)
    eris = cc.ao2mo(cc.mo_coeff)
    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    na, nb = noa*nva, nob*nvb

    t2 = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,nob,nvb,nvb))
    x, y = t2xy_2(t2)

    tl, bl = update_rpa(x, y, eris)
    tr, br = update_rpa(y, x, eris)
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
    print('check w split: {}'.format(np.linalg.norm(np.flip(w_minus)+w_plus)/w_minus.size))

    x, y = np.vsplit(v_plus, 2)
    t = np.dot(y, np.linalg.inv(x))
    print('T from diagonalization:\n{}'.format(t.real))
    t2aa = t[:na,:na].reshape((noa,nva,noa,nva)).transpose(0,2,1,3)
    t2ab = t[:na,na:].reshape((noa,nva,nob,nvb)).transpose(0,2,1,3)
    t2bb = t[na:,na:].reshape((nob,nvb,nob,nvb)).transpose(0,2,1,3)
    t1 = np.zeros((noa,nva)), np.zeros((nob,nvb))
    erccd = cc.energy(t1, (t2aa, t2ab, t2bb), eris) 
    print('max element: t2aa, t2ab, t2bb\n{}, {}, {}'.format(
          np.amax(abs(t2aa)),np.amax(abs(t2ab)),np.amax(abs(t2bb))))
    return w_plus, erccd
    

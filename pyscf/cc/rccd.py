import time, math, scipy, h5py
import numpy as np

from pyscf import lib, ao2mo, __config__
from pyscf.lib import logger
from pyscf.cc import ccsd, uccsd
from pyscf.ci import ucisd
from pyscf.fci.cistring import _gen_occslst

np.set_printoptions(precision=8,suppress=True)
einsum = lib.einsum
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def ci2occ(mf, ci, ndet):
    nea, neb = mf.mol.nelec 
    norba, norbb = mf.mo_coeff[0].shape[1], mf.mo_coeff[1].shape[1]
    occslsta = _gen_occslst(range(norba),nea)
    occslstb = _gen_occslst(range(norbb),neb)
    Na, Nb = occslsta.shape[0], occslstb.shape[0]
    assert ci.shape == (Na,Nb)
    ci = np.ravel(ci)
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

def c2t(c):
    ca, cb = c
    nmoa, noa = ca.shape
    nmob, nob = cb.shape
    nva, nvb = nmoa - noa, nmob - nob
    ta = np.dot(ca[noa:,:], np.linalg.inv(ca[:noa,:]))
    tb = np.dot(cb[nob:,:], np.linalg.inv(cb[:nob,:]))
    return (ta.T, tb.T)

def t2c(t):
    noa, nva = t[0].shape
    nob, nvb = t[1].shape
    Ia, Ib = np.eye(noa), np.eye(nob)
    ca, cb = np.vstack((Ia,t[0].T)), np.vstack((Ib,t[1].T))
    for k in range(noa):
        ca[:,k] /= np.linalg.norm(ca[:,k])
    for k in range(nob):
        cb[:,k] /= np.linalg.norm(cb[:,k])
    return (ca, cb)

def xy2t(x, y):
    noa, nva, nob, nvb = x[1].shape
    na, nb = noa*nva, nob*nvb
    xaa = x[0].reshape((na,na))
    xab = x[1].reshape((na,nb))
    xba = x[2].reshape((nb,na))
    xbb = x[3].reshape((nb,nb))
    yaa = y[0].reshape((na,na))
    yab = y[1].reshape((na,nb))
    yba = y[2].reshape((nb,na))
    ybb = y[3].reshape((nb,nb))
    x = np.block([[xaa,xab],[xba,xbb]])
    y = np.block([[yaa,yab],[yba,ybb]])
    t = np.dot(y, np.linalg.inv(x))
    taa = t[:na,:na].reshape((noa,nva,noa,nva)).transpose(0,2,1,3)
    tab = t[:na,na:].reshape((noa,nva,nob,nvb)).transpose(0,2,1,3)
    tbb = t[na:,na:].reshape((nob,nvb,nob,nvb)).transpose(0,2,1,3)
    return (taa, tab, tbb)

def t2xy(t):
    noa, nob, nva, nvb = t[1].shape
    yaa = t[0].transpose(0,2,1,3) 
    yab = t[1].transpose(0,2,1,3) 
    yba = t[1].transpose(1,3,0,2) 
    ybb = t[2].transpose(0,2,1,3) 
    xab, xba = np.zeros_like(yab), np.zeros_like(yba)
    Ioa, Iob = np.eye(noa), np.eye(nob)
    Iva, Ivb = np.eye(nva), np.eye(nvb)
    xaa = einsum('ij,ab->iajb',Ioa, Iva)
    xbb = einsum('ij,ab->iajb',Iob, Ivb)
    y = yaa, yab, yba, ybb
    x = xaa, xab, xba, xbb
    return x, y

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

def gs1(c):
    def gs(u):
        ncols = u.shape[1]
        v = u.copy()
        for k in range(1, ncols):
            for j in range(k):
                u[:,k] -= u[:,j]*np.dot(v[:,k],u[:,j])/np.dot(u[:,j],u[:,j])
        for k in range(ncols):
            u[:,k] /= np.linalg.norm(u[:,k])
        return u
    return (gs(c[0]), gs(c[1]))

def gs2(x, y):
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

def gs2_slow(x, y):
    xu, xv = x.copy(), x.copy()
    yu, yv = y.copy(), y.copy()
    ncols = x.shape[1]
    for k in range(1,ncols):
        for j in range(k):
            num = np.dot(xu[:,j],xv[:,k]) - np.dot(yu[:,j],yv[:,k])
            denom = np.dot(xu[:,j],xu[:,j]) - np.dot(yu[:,j],yu[:,j])
            xu[:,k] -= xu[:,j]*num/denom
            yu[:,k] -= yu[:,j]*num/denom
    return xu, yu

def update_ccs(t, eris):
    noa, nva = t[0].shape
    nob, nvb = t[1].shape
    fa, fb = eris.focka, eris.fockb
    ta, tb = t[0].T, t[1].T
    dta = - fa[noa:,:noa] - np.dot(fa[noa:,noa:],ta) + np.dot(ta,fa[:noa,:noa]) \
          + np.linalg.multi_dot([ta,fa[:noa,noa:],ta])
    dtb = - fb[nob:,:nob] - np.dot(fb[nob:,nob:],tb) + np.dot(tb,fb[:nob,:nob]) \
          + np.linalg.multi_dot([tb,fb[:nob,nob:],tb])
    return (dta.T, dtb.T)

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

    u2ab  = einsum('iack,kJcB->iJaB',eris.ovvo,t2ab)
    u2ab -= einsum('ikca,kJcB->iJaB',eris.oovv,t2ab)
    u2ab += einsum('iaCK,JKBC->iJaB',eris.ovVO,t2bb)
    u2ab += einsum('JBCK,iKaC->iJaB',eris.OVVO,t2ab)
    u2ab -= einsum('JKCB,iKaC->iJaB',eris.OOVV,t2ab)
    u2ab += einsum('JBck,ikac->iJaB',eris.OVvo,t2aa)
    u2ab += oOvV.copy()
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

def energy_slow(cc, t, eris):
    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    na, nb = noa*nva, nob*nvb
    taa = t[:na,:na].reshape((noa,nva,noa,nva)).transpose(0,2,1,3)
    tab = t[:na,na:].reshape((noa,nva,nob,nvb)).transpose(0,2,1,3)
    tbb = t[na:,na:].reshape((nob,nvb,nob,nvb)).transpose(0,2,1,3)
    t1 = np.zeros((noa, nva)), np.zeros((nob, nvb))
    return cc.energy(t1, (taa,tab,tbb), eris)

def kernel1(cc, beta, dt):

    e_exact = _ccs(cc._scf, cc.frozen)

    assert(cc.mo_coeff is not None)
    assert(cc.mo_occ is not None)
    if cc.verbose >= logger.WARN:
        ccsd.CCSD.check_sanity(cc)
    ccsd.CCSD.dump_flags(cc)
    log = logger.new_logger(cc, cc.verbose)

    eris = cc.ao2mo(cc.mo_coeff)
    fa, fb = eris.focka, eris.fockb

    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    t2 = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,nob,nvb,nvb))

    cr = np.eye(nmoa,noa), np.eye(nmob,nob)
    tr = c2t(cr)
    tq = tr[0].copy(), tr[1].copy() 
    cq = None
    quad = True

    cput1 = cput0 = (time.clock(), time.time())
    eold = 0
    conv = False
    er = cc.energy(tr, t2, eris)
    eq = cc.energy(tq, t2, eris)
    log.info('Init E(CCSr) = %.15g  E(CCSq) = %.15g', er, eq)

    ts = np.arange(0.0, beta, dt)
    tol = abs(e_exact) 
    tol_normt = cc.conv_tol_normt * dt
    for t in ts:
        # radon
        cr = cr[0]-np.dot(fa,cr[0])*dt, cr[1]-np.dot(fb,cr[1])*dt
        cr = gs1(cr)
        tr_new = c2t(cr)
        # quadratic
        if quad:
            dtq = update_ccs(tq, eris)
            tq_new = tq[0]+dtq[0]*dt, tq[1]+dtq[1]*dt
        else:
            if cq is None:
               cq = t2c(tq)
            cq = cq[0]-np.dot(fa,cq[0])*dt, cq[1]-np.dot(fb,cq[1])*dt
            cq = gs1(cq)
            tq_new = c2t(cq)
    
        normtr = np.linalg.norm(tr_new[0]-tr[0])+np.linalg.norm(tr_new[1]-tr[1])
        normtq = np.linalg.norm(tq_new[0]-tq[0])+np.linalg.norm(tq_new[1]-tq[1])
    
        tq, tq_new = tq_new, None
        tr, tr_new = tr_new, None
        eoldr, er = er, cc.energy(tr, t2, eris)
        eoldq, eq = eq, cc.energy(tq, t2, eris)
        log.info('t = %.4g  tq-tr = %.6g', t, diff)
        log.info('eq = %.15g  deq = %.9g  normtq = %.6g', 
                  eq, eq - eoldq, normtq)
        log.info('er = %.10g  der = %.8g  normtr = %.6g', 
                  er, er - eoldr, normtr)
    
        if normtq < tol_normt and normtr < tol_normt:
            break
        if abs(eq-eoldq) > tol:
            quad = False
        else:
            quad = True
            cq = None

    log.timer('CCSD', *cput0)

    cq = t2c(tq) 
    print('linear it coeff:\n{}\n{}'.format(cr[0], cr[1]))
    print('quadratic it coeff:\n{}\n{}'.format(cq[0], cq[1]))

    cc.converged = conv
    cc.e_corr = er
    cc.t1, cc.t2 = (tr[0].T,tr[1].T), t2
    ccsd.CCSD._finalize(cc)
    return 

def kernel2(cc, beta, dt):

    _, e_exact, (A,B) = _rpa(cc._scf, cc.frozen)
    exit()

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
    na, nb = noa*nva, nob*nvb
    t1 = np.zeros((noa, nva)), np.zeros((nob, nvb))

    tr = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,nob,nvb,nvb))
    xr, yr = t2xy(tr)
    xr_slow, yr_slow = np.eye(na+nb), np.zeros((na+nb,na+nb))
    tr_slow = yr_slow.copy()
    tq = tr[0].copy(), tr[1].copy(), tr[2].copy()
    xq = yq = None
    tq_slow = tr_slow.copy()
    xq_slow = yq_slow = None 

    cput1 = cput0 = (time.clock(), time.time())
    eoldq = eoldr = eoldq_slow = eoldr_slow = 0.0
    er = cc.energy(t1, tr, eris)
    eq = cc.energy(t1, tq, eris)
    eq_slow = energy_slow(cc, tq_slow, eris)  
    er_slow = energy_slow(cc, tr_slow, eris)  
    log.info('Init  E(CCDr) = %.15g  E(CCDq) = %.15g  E(CCDr slow) = %.15g  E(CCDq slow) = %.15g', er, eq, er_slow, eq_slow)

    ts = np.arange(0.0, beta, dt)
    ers = []
    eqs = []
    ers_slow = []
    eqs_slow = []
    tol = abs(e_exact) 
    tol_normt = cc.conv_tol_normt * dt
    quad = quad_slow = True
    conv = False
    for t in ts:
        # rpa
        dxr, dyr = update_rpa(xr, yr, eris)
        xr = xr[0]+dxr[0]*dt,xr[1]+dxr[1]*dt,xr[2]+dxr[2]*dt,xr[3]+dxr[3]*dt
        yr = yr[0]+dyr[0]*dt,yr[1]+dyr[1]*dt,yr[2]+dyr[2]*dt,yr[3]+dyr[3]*dt
        xr, yr = gs2(xr, yr)
        tr_new = xy2t(xr, yr)
        # rpa slow
        dxr_slow = np.dot(A,xr_slow) + np.dot(B,yr_slow)
        dyr_slow = np.dot(B,xr_slow) + np.dot(A,yr_slow)
        xr_slow, yr_slow = xr_slow + dt*dxr_slow, yr_slow - dt*dyr_slow
        xr_slow, yr_slow = gs2_slow(xr_slow, yr_slow)
        tr_slow_new = np.dot(yr_slow, np.linalg.inv(xr_slow))
        # rccd
        if quad:
            dtq = update_rccd(tq, eris)
            tq_new = tq[0]+dtq[0]*dt, tq[1]+dtq[1]*dt, tq[2]+dtq[2]*dt
        else:
            if xq is None:
               xq, yq = t2xy(tq)
            dxq, dyq = update_rpa(xq, yq, eris)
            xq = xq[0]+dxq[0]*dt,xq[1]+dxq[1]*dt,xq[2]+dxq[2]*dt,xq[3]+dxq[3]*dt
            yq = yq[0]+dyq[0]*dt,yq[1]+dyq[1]*dt,yq[2]+dyq[2]*dt,yq[3]+dyq[3]*dt
            xq, yq = gs2(xq, yq)
            tq_new = xy2t(xq, yq)
        # rccd slow
        if quad_slow:
            dtq_slow  = np.dot(A,tq_slow) + np.dot(tq_slow,A)
            dtq_slow += B + np.linalg.multi_dot([tq_slow,B,tq_slow])
            tq_slow_new = tq_slow - dtq_slow * dt
        else:
            if xq_slow is None:
                xq_slow, yq_slow = np.eye(na+nb), tq_slow.copy()
            dxq_slow = np.dot(A,xq_slow) + np.dot(B,yq_slow)
            dyq_slow = np.dot(B,xq_slow) + np.dot(A,yq_slow)
            xq_slow, yq_slow = xq_slow + dt*dxq_slow, yq_slow - dt*dyq_slow
            xq_slow, yq_slow = gs2_slow(xq_slow, yq_slow)
            tq_slow_new = np.dot(yq_slow, np.linalg.inv(xq_slow))
        
        diff  = np.linalg.norm(tq_new[0]-tr_new[0]) 
        diff += np.linalg.norm(tq_new[1]-tr_new[1]) 
        diff += np.linalg.norm(tq_new[2]-tr_new[2]) 
        diff_slow = np.linalg.norm(tq_slow_new - tr_slow_new)
    
        normtq = np.linalg.norm(cc.amplitudes_to_vector(t1, tq_new) -
                                cc.amplitudes_to_vector(t1, tq))
        normtr = np.linalg.norm(cc.amplitudes_to_vector(t1, tr_new) -
                                cc.amplitudes_to_vector(t1, tr))
        normtq_slow = np.linalg.norm(tq_slow_new - tq_slow)
        normtr_slow = np.linalg.norm(tr_slow_new - tr_slow)

        tq, tq_new = tq_new, None
        tr, tr_new = tr_new, None
        tq_slow, tq_slow_new = tq_slow_new, None
        tr_slow, tr_slow_new = tr_slow_new, None

        eoldq, eq = eq, cc.energy(t1, tq, eris)
        eoldr, er = er, cc.energy(t1, tr, eris)
        eoldq_slow, eq_slow = eq_slow, energy_slow(cc, tq_slow, eris)
        eoldr_slow, er_slow = er_slow, energy_slow(cc, tr_slow, eris)

        eqs.append(eq - e_exact)
        ers.append(er - e_exact)
        eqs_slow.append(eq_slow - e_exact)
        ers_slow.append(er_slow - e_exact)

        log.info('t = %.4g  tq-tr = %.6g  tq-tr(slow) = %.6g', t, diff, diff_slow)
        log.info('     eq = %.15g  deq = %.9g  normtq = %.6g', eq, eq - eoldq, normtq)
        log.info('slow eq = %.15g  deq = %.9g  normtq = %.6g', 
                 eq_slow, eq_slow - eoldq_slow, normtq_slow)
        log.info('     er = %.10g  der = %.8g  normtr = %.6g', er, er - eoldr, normtr)
        log.info('slow er = %.10g  der = %.8g  normtr = %.6g', 
                 er_slow, er_slow - eoldr_slow, normtr_slow)
        conv = normtq < tol_normt and normtr < tol_normt and \
               normtq < tol_normt and normtr < tol_normt
        if conv:
            break
        if abs(eq - eoldq) > tol:
            quad = False
        else:
            quad = True
            xq = yq = None
        if abs(eq_slow - eoldq_slow) > tol:
            quad_slow = False
        else:
            quad_slow = True
            xq_slow = yq_slow = None

    log.timer('CCSD', *cput0)

    t2aa, t2ab, t2bb = tr
    taa = t2aa.transpose(0,2,1,3).reshape((na,na))
    tab = t2ab.transpose(0,2,1,3).reshape((na,nb))
    tba = t2ab.transpose(1,3,0,2).reshape((nb,na))
    tbb = t2bb.transpose(0,2,1,3).reshape((nb,nb))
    print('rpa T from imaginary time:\n{}'.format(np.block([[taa,tab],[tba,tbb]])))

    t2aa, t2ab, t2bb = tq
    taa = t2aa.transpose(0,2,1,3).reshape((na,na))
    tab = t2ab.transpose(0,2,1,3).reshape((na,nb))
    tba = t2ab.transpose(1,3,0,2).reshape((nb,na))
    tbb = t2bb.transpose(0,2,1,3).reshape((nb,nb))
    print('rccd T from imaginary time:\n{}'.format(np.block([[taa,tab],[tba,tbb]])))

    print('slow rpa T from imaginary time:\n{}'.format(tr_slow))
    print('slow rccd T from imaginary time:\n{}'.format(tq_slow))

    cc.converged = conv
    cc.e_corr = er
    cc.t1, cc.t2 = t1, tr
    ccsd.CCSD._finalize(cc)
    print('max element rpa: t2aa, t2ab, t2bb\n{}, {}, {}'.format(
          np.amax(abs(tr[0])),np.amax(abs(tr[1])),np.amax(abs(tr[2]))))
    print('max element rccd: t2aa, t2ab, t2bb\n{}, {}, {}'.format(
          np.amax(abs(tq[0])),np.amax(abs(tq[1])),np.amax(abs(tq[2]))))
    return ts[:len(eqs)], np.array(eqs), np.array(ers), np.array(eqs_slow), np.array(ers_slow)

class itCCSD(ccsd.CCSD):

    conv_tol = getattr(__config__, 'cc_uccsd_UCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_uccsd_UCCSD_conv_tol_normt', 1e-6)

# Attribute frozen can be
# * An integer : The same number of inner-most alpha and beta orbitals are frozen
# * One list : Same alpha and beta orbital indices to be frozen
# * A pair of list : First list is the orbital indices to be frozen for alpha
#       orbitals, second list is for beta orbitals
    def __init__(self, mf, ci=None, ndet=None, frozen=0, beta=10.0, dt=0.1):

        self.cc = uccsd.UCCSD(mf, frozen)
        self.mo0 = mf.mo_coeff[1].copy(), mf.mo_coeff[1].copy()

        self.occ = ci2occ(mf, ci, ndet)
        ndet = self.occ.shape[0]
        self.beta = beta
        self.dt = dt

    def kernel1(self):
        ndet = self.occ.shape[0]
        eccsd = np.zeros(ndet)
        for I in range(ndet):
            print('\nreference {} occupation: {}'.format(I, self.occ[I,:]))
            mo_coeff = perm_mo(self.cc._scf, self.mo0, self.occ[I,:])
            self.cc.mo_coeff = self.cc._scf.mo_coeff = mo_coeff
            e_ref = ref_energy(self.cc._scf)
            self.cc._scf.e_tot = e_ref + self.cc._scf.mol.energy_nuc()
            print('reference energy: {}'.format(self.cc._scf.e_tot))
            kernel1(self.cc, self.beta, self.dt)
        return 

    def kernel2(self):
        ndet = self.occ.shape[0]
        eccsd = np.zeros(ndet)
        for I in range(ndet):
            print('\nreference {} occupation: {}'.format(I, self.occ[I,:]))
            mo_coeff = perm_mo(self.cc._scf, self.mo0, self.occ[I,:])
            self.cc.mo_coeff = self.cc._scf.mo_coeff = mo_coeff
            e_ref = ref_energy(self.cc._scf)
            self.cc._scf.e_tot = e_ref + self.cc._scf.mol.energy_nuc()
            print('reference energy: {}'.format(self.cc._scf.e_tot))
            t, eq, er, eq_slow, er_slow = kernel2(self.cc, self.beta, self.dt)
            f = h5py.File('{}.hdf5'.format(I), 'w')
            f['t'] = t.copy()
            f['eq'] = eq.copy()
            f['er'] = er.copy()
            f['eq_slow'] = eq_slow.copy()
            f['er_slow'] = er_slow.copy()
            f.close()
        return 

def _ccs(mf, frozen):
    cc = uccsd.UCCSD(mf, frozen)
    eris = cc.ao2mo(cc.mo_coeff)
    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    
    wa, va = np.linalg.eigh(eris.focka)
    wb, vb = np.linalg.eigh(eris.fockb)
    print('diagonalized coeff: \n{}\n{}'.format(va[:,:noa],vb[:,:noa]))

    try:
        t1a = np.dot(va[noa:,:noa],np.linalg.inv(va[:noa,:noa])).T
        t1b = np.dot(vb[nob:,:nob],np.linalg.inv(vb[:nob,:nob])).T
    except:
        t1a, t1b = np.zeros((noa,nva)), np.zeros((nob,nvb))
    t2 = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,nob,nvb,nvb))
    eccs = cc.energy((t1a,t1b), t2, eris)
    print('eccs: {}'.format(eccs))
    return eccs

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
    x, y = t2xy(t2)

    tl, bl = update_rpa(x, y, eris)
    tlaa, tlab, tlba, tlbb = tl
    blaa, blab, blba, blbb = bl
#
    eoa, eva = eris.focka.diagonal()[:noa], eris.focka.diagonal()[noa:]
    eob, evb = eris.focka.diagonal()[:nob], eris.focka.diagonal()[nob:]
    eia_a = eva[None,:,None,None] - eoa[:,None,None,None] 
    eia_b = evb[None,:,None,None] - eob[:,None,None,None]
    Ioa, Iob = np.eye(noa), np.eye(nob)
    Iva, Ivb = np.eye(nva), np.eye(nvb)
    Aaa  = einsum('ij,ab->iajb',Ioa, Iva)*eia_a
    Aaa += eris.ovvo.transpose(0,1,3,2) - eris.oovv.transpose(0,3,1,2)
    Aab  = eris.ovVO.transpose(0,1,3,2)
    Aba  = eris.ovVO.transpose(3,2,0,1)
    Abb  = einsum('ij,ab->iajb',Iob, Ivb)*eia_b
    Abb += eris.OVVO.transpose(0,1,3,2) - eris.OOVV.transpose(0,3,1,2)
    Baa  = eris.ovov - eris.ovov.transpose(0,3,2,1)
    Bab  = eris.ovOV
    Bba  = eris.ovOV.transpose(2,3,0,1)
    Bbb  = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    diff  = np.linalg.norm(Aaa-tlaa)
    diff += np.linalg.norm(Aab-tlab)
    diff += np.linalg.norm(Aba-tlba)
    diff += np.linalg.norm(Abb-tlbb)
    diff += np.linalg.norm(Baa+blaa)
    diff += np.linalg.norm(Bab+blab)
    diff += np.linalg.norm(Bba+blba)
    diff += np.linalg.norm(Bbb+blbb)
    print('AB fomration: {}'.format(diff))
#
    tlaa = tlaa.reshape((na,na))
    tlab = tlab.reshape((na,nb))
    tlba = tlba.reshape((nb,na))
    tlbb = tlbb.reshape((nb,nb))
    blaa = blaa.reshape((na,na))
    blab = blab.reshape((na,nb))
    blba = blba.reshape((nb,na))
    blbb = blbb.reshape((nb,nb))
    
    tl = np.block([[tlaa,tlab],[tlba,tlbb]])
    bl = np.block([[blaa,blab],[blba,blbb]])

    M = np.block([[tl,-bl],[bl,-tl]])
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
    erccd = cc.energy(t1, (t2aa, t2ab, t2bb), eris) 
    print('max element: t2aa, t2ab, t2bb\n{}, {}, {}'.format(
          np.amax(abs(t2aa)),np.amax(abs(t2ab)),np.amax(abs(t2bb))))
    print('erccd: {}'.format(erccd))
    print('excitation energy (eV):\n{}'.format(w_plus.real*27.2114))
    return w_plus, erccd, (tl, -bl)
    

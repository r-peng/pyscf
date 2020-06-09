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
    tba = t[na:,:na].reshape((nob,nvb,noa,nva)).transpose(0,2,1,3)
    tbb = t[na:,na:].reshape((nob,nvb,nob,nvb)).transpose(0,2,1,3)
    return (taa, tab, tba, tbb)

def t2xy(t):
    noa, nob, nva, nvb = t[1].shape
    yaa = t[0].transpose(0,2,1,3) 
    yab = t[1].transpose(0,2,1,3) 
    yba = t[2].transpose(0,2,1,3) 
    ybb = t[3].transpose(0,2,1,3) 
    xab, xba = np.zeros_like(yab), np.zeros_like(yba)
    Ioa, Iob = np.eye(noa), np.eye(nob)
    Iva, Ivb = np.eye(nva), np.eye(nvb)
    xaa = einsum('ij,ab->iajb',Ioa, Iva)
    xbb = einsum('ij,ab->iajb',Iob, Ivb)
    y = yaa, yab, yba, ybb
    x = xaa, xab, xba, xbb
    return x, y

def getAB(eris):
    noa, nva, nob, nvb = eris.ovOV.shape
    na, nb = noa*nva, nob*nvb

    eoa, eva = eris.focka.diagonal()[:noa], eris.focka.diagonal()[noa:]
    eob, evb = eris.focka.diagonal()[:nob], eris.focka.diagonal()[nob:]
    eia_a = eva[None,:,None,None] - eoa[:,None,None,None] 
    eia_b = evb[None,:,None,None] - eob[:,None,None,None]
    Ioa, Iob = np.eye(noa), np.eye(nob)
    Iva, Ivb = np.eye(nva), np.eye(nvb)

    Aaa  = einsum('ij,ab->iajb',Ioa, Iva)*eia_a
    Aaa += eris.ovvo.transpose(0,1,3,2) - eris.oovv.transpose(0,3,1,2)
    Aab  = eris.ovVO.transpose(0,1,3,2)
    Abb  = einsum('ij,ab->iajb',Iob, Ivb)*eia_b
    Abb += eris.OVVO.transpose(0,1,3,2) - eris.OOVV.transpose(0,3,1,2)
    Baa  = eris.ovov - eris.ovov.transpose(0,3,2,1)
    Bab  = eris.ovOV
    Bbb  = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    Aop, Bop = (Aaa, Aab, Abb), (Baa, Bab, Bbb)

    Aaa = Aaa.reshape((na,na))
    Aab = Aab.reshape((na,nb))
    Abb = Abb.reshape((nb,nb))
    Baa = Baa.reshape((na,na))
    Bab = Bab.reshape((na,nb))
    Bbb = Bbb.reshape((nb,nb))
    A = np.block([[Aaa,Aab],[Aab.T,Abb]])
    B = np.block([[Baa,Bab],[Bab.T,Bbb]])
    return Aop, Bop, A, B

def update_rpa(x, y, A, B):
    (Aaa, Aab, Abb), (Baa, Bab, Bbb) = A, B
    (xaa, xab, xba, xbb), (yaa, yab,yba, ybb) = x, y
    dxaa  = einsum('iakc,kcjb->iajb',Aaa,xaa) + einsum('iaKC,KCjb->iajb',Aab,xba)
    dxaa += einsum('iakc,kcjb->iajb',Baa,yaa) + einsum('iaKC,KCjb->iajb',Bab,yba)
    dxab  = einsum('iakc,kcJB->iaJB',Aaa,xab) + einsum('iaKC,KCJB->iaJB',Aab,xbb)
    dxab += einsum('iakc,kcJB->iaJB',Baa,yab) + einsum('iaKC,KCJB->iaJB',Bab,ybb)
    dxba  = einsum('kcIA,kcjb->IAjb',Aab,xaa) + einsum('IAKC,KCjb->IAjb',Abb,xba)
    dxba += einsum('kcIA,kcjb->IAjb',Bab,yaa) + einsum('IAKC,KCjb->IAjb',Bbb,yba)
    dxbb  = einsum('kcIA,kcJB->IAJB',Aab,xab) + einsum('IAKC,KCJB->IAJB',Abb,xbb)
    dxbb += einsum('kcIA,kcJB->IAJB',Bab,yab) + einsum('IAKC,KCJB->IAJB',Bbb,ybb)

    dyaa  = einsum('iakc,kcjb->iajb',Aaa,yaa) + einsum('iaKC,KCjb->iajb',Aab,yba)
    dyaa += einsum('iakc,kcjb->iajb',Baa,xaa) + einsum('iaKC,KCjb->iajb',Bab,xba)
    dyab  = einsum('iakc,kcJB->iaJB',Aaa,yab) + einsum('iaKC,KCJB->iaJB',Aab,ybb)
    dyab += einsum('iakc,kcJB->iaJB',Baa,xab) + einsum('iaKC,KCJB->iaJB',Bab,xbb)
    dyba  = einsum('kcIA,kcjb->IAjb',Aab,yaa) + einsum('IAKC,KCjb->IAjb',Abb,yba)
    dyba += einsum('kcIA,kcjb->IAjb',Bab,xaa) + einsum('IAKC,KCjb->IAjb',Bbb,xba)
    dybb  = einsum('kcIA,kcJB->IAJB',Aab,yab) + einsum('IAKC,KCJB->IAJB',Abb,ybb)
    dybb += einsum('kcIA,kcJB->IAJB',Bab,xab) + einsum('IAKC,KCJB->IAJB',Bbb,xbb)
    return (dxaa, dxab, dxba, dxbb), (-dyaa, -dyab, -dyba, -dybb)

def gs2(x, y):
    def gs(x, y):
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
    tup = True if type(x) is tuple else False
    if tup:
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
        x, y = np.block([[xaa,xab],[xba,xbb]]), np.block([[yaa,yab],[yba,ybb]])
    x, y = gs(x, y)
    if tup: 
        xaa = x[:na,:na].reshape((noa,nva,noa,nva))
        xab = x[:na,na:].reshape((noa,nva,nob,nvb))
        xba = x[na:,:na].reshape((nob,nvb,noa,nva))
        xbb = x[na:,na:].reshape((nob,nvb,nob,nvb))
        yaa = y[:na,:na].reshape((noa,nva,noa,nva))
        yab = y[:na,na:].reshape((noa,nva,nob,nvb))
        yba = y[na:,:na].reshape((nob,nvb,noa,nva))
        ybb = y[na:,na:].reshape((nob,nvb,nob,nvb))
        x, y = (xaa,xab,xba,xbb), (yaa,yab,yba,ybb)
    return x, y

def update_rccd(t, A, B):
    taa, tab, tba, tbb = t
    taa = taa.transpose(0,2,1,3)
    tab = tab.transpose(0,2,1,3)
    tba = tba.transpose(0,2,1,3)
    tbb = tbb.transpose(0,2,1,3)
    (Aaa, Aab, Abb), (Baa, Bab, Bbb) = A, B
    uaa  = einsum('iakc,kcjb->iajb',Aaa,taa) + einsum('iaKC,KCjb->iajb',Aab,tba)
    uaa += einsum('iakc,kcjb->iajb',taa,Aaa) + einsum('iaKC,jbKC->iajb',tab,Aab)
    uaa += Baa.copy()
    temp = einsum('kcld,ldjb->kcjb',Baa,taa) + einsum('kcLD,LDjb->kcjb',Bab,tba)
    uaa += einsum('iakc,kcjb->iajb',taa,temp)
    temp = einsum('ldKC,ldjb->KCjb',Bab,taa) + einsum('KCLD,LDjb->KCjb',Bbb,tba)
    uaa += einsum('iaKC,KCjb->iajb',tab,temp)

    ubb  = einsum('kcIA,kcJB->IAJB',Aab,tab) + einsum('IAKC,KCJB->IAJB',Abb,tbb)
    ubb += einsum('IAkc,kcJB->IAJB',tba,Aab) + einsum('IAKC,KCJB->IAJB',tbb,Abb)
    ubb += Bbb.copy()
    temp = einsum('kcld,ldJB->kcJB',Baa,tab) + einsum('kcLD,LDJB->kcJB',Bab,tbb)
    ubb += einsum('IAkc,kcJB->IAJB',tba,temp)
    temp = einsum('ldKC,ldJB->KCJB',Bab,tab) + einsum('KCLD,LDJB->KCJB',Bbb,tbb)
    ubb += einsum('IAKC,KCJB->IAJB',tbb,temp)

    uab  = einsum('iakc,kcJB->iaJB',Aaa,tab) + einsum('iaKC,KCJB->iaJB',Aab,tbb)
    uab += einsum('iakc,kcJB->iaJB',taa,Aab) + einsum('iaKC,KCJB->iaJB',tab,Abb)
    uab += Bab.copy()
    temp = einsum('kcld,ldJB->kcJB',Baa,tab) + einsum('kcLD,LDJB->kcJB',Bab,tbb)
    uab += einsum('iakc,kcJB->iaJB',taa,temp)
    temp = einsum('ldKC,ldJB->KCJB',Bab,tab) + einsum('KCLD,LDJB->KCJB',Bbb,tbb)
    uab += einsum('iaKC,KCJB->iaJB',tab,temp)

    uba  = einsum('kcIA,kcjb->IAjb',Aab,taa) + einsum('IAKC,KCjb->IAjb',Abb,tba)
    uba += einsum('IAkc,kcjb->IAjb',tba,Aaa) + einsum('IAKC,jbKC->IAjb',tbb,Aab)
    uba += Bab.transpose(2,3,0,1)
    temp = einsum('kcld,ldjb->kcjb',Baa,taa) + einsum('kcLD,LDjb->kcjb',Bab,tba)
    uba += einsum('IAkc,kcjb->IAjb',tba,temp)
    temp = einsum('ldKC,ldjb->KCjb',Bab,taa) + einsum('KCLD,LDjb->KCjb',Bbb,tba)
    uba += einsum('IAKC,KCjb->IAjb',tbb,temp)

    uaa = -uaa.transpose(0,2,1,3)
    uab = -uab.transpose(0,2,1,3)
    uba = -uba.transpose(0,2,1,3)
    ubb = -ubb.transpose(0,2,1,3)
    return (uaa, uab, uba, ubb)

def energy(t, B):
    if type(t) is tuple:
        noa, nob, nva, nvb = t[1].shape
        na, nb = noa*nva, nob*nvb
        taa = t[0].transpose(0,2,1,3).reshape(na,na)
        tab = t[1].transpose(0,2,1,3).reshape(na,nb)
        tba = t[2].transpose(0,2,1,3).reshape(nb,na)
        tbb = t[3].transpose(0,2,1,3).reshape(nb,nb)
        t = np.block([[taa,tab],[tba,tbb]])
    return 0.5*einsum('IJ,IJ',B,t)

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


    assert(cc.mo_coeff is not None)
    assert(cc.mo_occ is not None)
    if cc.verbose >= logger.WARN:
        ccsd.CCSD.check_sanity(cc)
    ccsd.CCSD.dump_flags(cc)
    log = logger.new_logger(cc, cc.verbose)

    eris = cc.ao2mo(cc.mo_coeff)
    Aop, Bop, A, B = getAB(eris)
    _, e_exact = _rpa(cc._scf, cc.frozen, A, B)

    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    nva, nvb = nmoa - noa, nmob - nob
    na, nb = noa*nva, nob*nvb
    t1 = np.zeros((noa, nva)), np.zeros((nob, nvb))

    tr = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,noa,nvb,nva)), np.zeros((nob,nob,nvb,nvb))
    tq = np.zeros((noa,noa,nva,nva)), np.zeros((noa,nob,nva,nvb)), \
         np.zeros((nob,noa,nvb,nva)), np.zeros((nob,nob,nvb,nvb))
    tr_slow = np.zeros((na+nb,na+nb))
    tq_slow = np.zeros((na+nb,na+nb))
    xr, yr = t2xy(tr)
    xr_slow, yr_slow = np.eye(na+nb), tr_slow.copy() 
    xq = yq = xq_slow = yq_slow = None 

    cput1 = cput0 = (time.clock(), time.time())
    eoldq = eoldr = eoldq_slow = eoldr_slow = 0.0
    er = energy(tr, B)
    eq = energy(tq, B)
    eq_slow = energy(tq_slow, B)  
    er_slow = energy(tr_slow, B)  
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
        dx, dy = update_rpa(xr, yr, Aop, Bop)
        xr = xr[0]+dx[0]*dt,xr[1]+dx[1]*dt,xr[2]+dx[2]*dt,xr[3]+dx[3]*dt
        yr = yr[0]+dy[0]*dt,yr[1]+dy[1]*dt,yr[2]+dy[2]*dt,yr[3]+dy[3]*dt
        xr, yr = gs2(xr, yr)
        tr_new = xy2t(xr, yr)
        # rpa slow
        dx = np.dot(A,xr_slow) + np.dot(B,yr_slow)
        dy = np.dot(B,xr_slow) + np.dot(A,yr_slow)
        xr_slow, yr_slow = xr_slow + dt*dx, yr_slow - dt*dy
        xr_slow, yr_slow = gs2(xr_slow, yr_slow)
        tr_slow_new = np.dot(yr_slow, np.linalg.inv(xr_slow))
        # rccd
        if quad:
            u = update_rccd(tq, Aop, Bop)
            tq_new = tq[0]+u[0]*dt,tq[1]+u[1]*dt,tq[2]+u[2]*dt,tq[3]+u[3]*dt
        else:
            print('rpa')
            if xq is None:
               xq, yq = t2xy(tq)
            dx, dy = update_rpa(xq, yq, Aop, Bop)
            xq = xq[0]+dx[0]*dt,xq[1]+dx[1]*dt,xq[2]+dx[2]*dt,xq[3]+dx[3]*dt
            yq = yq[0]+dy[0]*dt,yq[1]+dy[1]*dt,yq[2]+dy[2]*dt,yq[3]+dy[3]*dt
            xq, yq = gs2(xq, yq)
            tq_new = xy2t(xq, yq)
        # rccd slow
        if quad_slow:
            u  = np.dot(A,tq_slow) + np.dot(tq_slow,A)
            u += B + np.linalg.multi_dot([tq_slow,B,tq_slow])
            tq_slow_new = tq_slow - u * dt
        else:
            print('rpa_slow')
            if xq_slow is None:
                xq_slow, yq_slow = np.eye(na+nb), tq_slow.copy()
                xq_slow, yq_slow = gs2(xq_slow, yq_slow)
            dx = np.dot(A,xq_slow) + np.dot(B,yq_slow)
            dy = np.dot(B,xq_slow) + np.dot(A,yq_slow)
            xq_slow, yq_slow = xq_slow + dt*dx, yq_slow - dt*dy
            xq_slow, yq_slow = gs2(xq_slow, yq_slow)
            tq_slow_new = np.dot(yq_slow, np.linalg.inv(xq_slow))
        
        taa = tq_new[0].transpose(0,2,1,3).reshape(na,na)
        tab = tq_new[1].transpose(0,2,1,3).reshape(na,nb)
        tba = tq_new[2].transpose(0,2,1,3).reshape(nb,na)
        tbb = tq_new[3].transpose(0,2,1,3).reshape(nb,nb)
        diffq = np.linalg.norm(tq_slow_new-np.block([[taa,tab],[tba,tbb]]))   
        if diffq > 1e-6:
            break
        taa = tr_new[0].transpose(0,2,1,3).reshape(na,na)
        tab = tr_new[1].transpose(0,2,1,3).reshape(na,nb)
        tba = tr_new[2].transpose(0,2,1,3).reshape(nb,na)
        tbb = tr_new[3].transpose(0,2,1,3).reshape(nb,nb)
        diffr = np.linalg.norm(tr_slow_new-np.block([[taa,tab],[tba,tbb]]))   

        normtq = np.linalg.norm(cc.amplitudes_to_vector(t1, tq_new) -
                                cc.amplitudes_to_vector(t1, tq))
        normtr = np.linalg.norm(cc.amplitudes_to_vector(t1, tr_new) -
                                cc.amplitudes_to_vector(t1, tr))
        normtq_slow = np.linalg.norm(tq_slow_new - tq_slow)
        normtr_slow = np.linalg.norm(tr_slow_new - tr_slow)

        tq, tq_new = tq_new, None
        tr, tr_new = tr_new, None
        tq_slow, tq_slow_new = tq_slow_new.copy(), None
        tr_slow, tr_slow_new = tr_slow_new.copy(), None

        eoldq, eq = eq, energy(tq, B)
        eoldr, er = er, energy(tr, B)
        eoldq_slow, eq_slow = eq_slow, energy(tq_slow, B)
        eoldr_slow, er_slow = er_slow, energy(tr_slow, B)

        eqs.append(eq - e_exact)
        ers.append(er - e_exact)
        eqs_slow.append(eq_slow - e_exact)
        ers_slow.append(er_slow - e_exact)

        log.info('t = %.4g  tq-tq_slow = %.6g  tr-tr_slow = %.6g', t, diffq, diffr)
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

    taa, tab, tba, tbb = tr
    taa = taa.transpose(0,2,1,3).reshape((na,na))
    tab = tab.transpose(0,2,1,3).reshape((na,nb))
    tba = tba.transpose(0,2,1,3).reshape((nb,na))
    tbb = tbb.transpose(0,2,1,3).reshape((nb,nb))
    print('rpa T from imaginary time:\n{}'.format(np.block([[taa,tab],[tba,tbb]])))

    taa, tab, tba, tbb = tq
    taa = taa.transpose(0,2,1,3).reshape((na,na))
    tab = tab.transpose(0,2,1,3).reshape((na,nb))
    tba = tba.transpose(0,2,1,3).reshape((nb,na))
    tbb = tbb.transpose(0,2,1,3).reshape((nb,nb))
    print('rccd T from imaginary time:\n{}'.format(np.block([[taa,tab],[tba,tbb]])))

    print('slow rpa T from imaginary time:\n{}'.format(tr_slow))
    print('slow rccd T from imaginary time:\n{}'.format(tq_slow))

    cc.converged = conv
    cc.e_corr = er
    cc.t1, cc.t2 = t1, tr
    ccsd.CCSD._finalize(cc)
    print('max element rpa: taa, tab, tbb\n{}, {}, {}'.format(
          np.amax(abs(tr[0])),np.amax(abs(tr[1])),np.amax(abs(tr[3]))))
    print('max element rccd: taa, tab, tbb\n{}, {}, {}'.format(
          np.amax(abs(tq[0])),np.amax(abs(tq[1])),np.amax(abs(tq[3]))))
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

def _rpa(mf, frozen=0, A=None, B=None):
    cc = uccsd.UCCSD(mf, frozen)
    noa, nob = cc.get_nocc()
    nmoa, nmob = cc.get_nmo()
    eris = cc.ao2mo(cc.mo_coeff)
    nva, nvb = nmoa - noa, nmob - nob
    na, nb = noa*nva, nob*nvb

    if A is None:
        _, _, A, B = getAB(eris)
    w, v = scipy.linalg.eig(np.block([[A,B],[-B,-A]]))

    idx = np.argsort(w.real) 
    w = w[idx]
    v = v[:,idx]
    w_minus, w_plus = np.hsplit(w, 2)
    v_minus, v_plus = np.hsplit(v, 2)
    print('check w split: {}'.format(np.linalg.norm(np.flip(w_minus)+w_plus)/w_minus.size))

    x, y = np.vsplit(v_plus.real, 2)
    t = np.dot(y, np.linalg.inv(x))
    erccd = energy(t, B) 
    print('T from diagonalization:\n{}'.format(t.real))
    print('max element: taa, tab, tbb\n{}, {}, {}'.format(
    np.amax(abs(t[:na,:na])),np.amax(abs(t[:na,na:])),np.amax(abs(t[na:,na:]))))
    print('excitation energy (eV):\n{}'.format(w_plus.real*27.2114))
    print('erccd: {}'.format(erccd))

    taa = t[:na,:na].reshape((noa,nva,noa,nva)).transpose(0,2,1,3)
    tab = t[:na,na:].reshape((noa,nva,nob,nvb)).transpose(0,2,1,3)
    tbb = t[na:,na:].reshape((nob,nvb,nob,nvb)).transpose(0,2,1,3)
    print(np.linalg.norm(taa-taa.transpose(1,0,3,2)))
    print(0.5*np.trace(np.diag(w_plus.real)-A))

    return w_plus, erccd
    

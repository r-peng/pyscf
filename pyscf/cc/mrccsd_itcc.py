import numpy
import time
from pyscf import lib
from pyscf.cc import uccsd
from cqcpy import cc_energy, cc_equations
from cqcpy.ov_blocks import make_two_e_blocks, make_two_e_blocks_full
from kelvin import zt_mp, cc_utils, quadrature, propagation, td_ccsd 
from kelvin import scf_system, scf_utils

einsum = lib.einsum
#einsum = numpy.einsum

class ccsd(object):
    def __init__(self, mf, singles=True, 
        beta=20, ngrid=50, quad='lin'):

        sys = scf_system.scf_system(mf, 0, 0)
        self.sys = sys
        self.beta = beta
        self.nfc = 0,0 
        self.nfv = 0,0 
        self.singles = singles
        self.ngrid = ngrid
        ng = self.ngrid
        self.ti, self.g,  _ = quadrature.ft_quad(ngrid, beta, quad)
        self.iprint = mf.mol.verbose

def _uccsd(cc,T1in=None,T2in=None,denom=None):
    """Solve finite temperature coupled cluster equations."""
    # get time-grid
    ng = cc.ngrid
    ti = cc.ti
    g = cc.g
    beta = cc.beta

    # get Fock matrix
    Fa,Fb = cc.sys.u_fock()
    eoa = numpy.diagonal(Fa.oo)
    eva = numpy.diagonal(Fa.vv)
    eob = numpy.diagonal(Fb.oo)
    evb = numpy.diagonal(Fb.vv)

    # get HF energy
    En = cc.sys.const_energy()
    E0 = zt_mp.ump0(eoa, eob) + En
    E1 = cc.sys.get_mp1()
    E01 = E0 + E1

    # get ERIs
    Ia, Ib, Iabab = cc.sys.u_aint_tot()

    # get frozen tensors
    nmoa, nmob = Iabab.shape[:2]
    nfca, nfcb = cc.nfc
    nfva, nfvb = cc.nfv
    noa, nva, nob, nvb = eoa.size-nfca, eva.size-nfva, eob.size-nfcb, evb.size-nfvb
    Fa.oo = Fa.oo[nfca:,nfca:] 
    Fb.oo = Fb.oo[nfcb:,nfcb:]
    Fa.ov = Fa.ov[nfca:,:nva] 
    Fb.ov = Fb.ov[nfcb:,:nvb] 
    Fa.vo = Fa.vo[:nva,nfca:] 
    Fb.vo = Fb.vo[:nvb,nfcb:] 
    Fa.vv = Fa.vv[:nva,:nva]
    Fb.vv = Fb.vv[:nvb,:nvb]
    if denom is not None:
        eoa, eob, eva, evb = denom
    else:
        eoa, eob, eva, evb = eoa[nfca:],eob[nfcb:],eva[:nva],evb[:nvb] 
    Fa.oo = Fa.oo - numpy.diag(eoa) # subtract diagonal
    Fa.vv = Fa.vv - numpy.diag(eva) # subtract diagonal
    Fb.oo = Fb.oo - numpy.diag(eob) # subtract diagonal
    Fb.vv = Fb.vv - numpy.diag(evb) # subtract diagonal

    Ia = Ia[nfca:nmoa-nfva,nfca:nmoa-nfva,nfca:nmoa-nfva,nfca:nmoa-nfva]
    Ib = Ib[nfcb:nmob-nfvb,nfcb:nmob-nfvb,nfcb:nmob-nfvb,nfcb:nmob-nfvb]
    Iabab = Iabab[nfca:nmoa-nfva,nfcb:nmob-nfvb,nfca:nmoa-nfva,nfcb:nmob-nfvb]
    Ia = make_two_e_blocks(Ia,noa,nva,noa,nva,noa,nva,noa,nva)
    Ib = make_two_e_blocks(Ib,nob,nvb,nob,nvb,nob,nvb,nob,nvb)
    Iabab = make_two_e_blocks_full(Iabab,noa,nva,nob,nvb,noa,nva,nob,nvb)
    D1a = (eva[:,None] - eoa[None,:])
    D1b = (evb[:,None] - eob[None,:])
    D2aa = (eva[:,None,None,None] + eva[None,:,None,None]
        - eoa[None,None,:,None] - eoa[None,None,None,:])
    D2ab = (eva[:,None,None,None] + evb[None,:nvb,None,None]
        - eoa[None,None,:,None] - eob[None,None,None,:])
    D2bb = (evb[:,None,None,None] + evb[None,:,None,None]
        - eob[None,None,:,None] - eob[None,None,None,:])

    t1a = numpy.zeros((nva,noa), dtype=Fa.vo.dtype)
    t1b = numpy.zeros((nvb,nob), dtype=Fa.vo.dtype)
    t2aa = numpy.zeros((nva,nva,noa,noa), dtype=Ia.vvoo.dtype)
    t2ab = numpy.zeros((nva,nvb,noa,nob), dtype=Iabab.vvoo.dtype)
    t2bb = numpy.zeros((nvb,nvb,nob,nob), dtype=Ib.vvoo.dtype)
    Eccn = 0.0

    def fRHS(var):
        t1a,t1b,t2aa,t2ab,t2bb = var
        k1sa = -D1a*t1a - Fa.vo.copy()
        k1sb = -D1b*t1b - Fb.vo.copy()
        k1daa = -D2aa*t2aa - Ia.vvoo.copy()
        k1dab = -D2ab*t2ab - Iabab.vvoo.copy()
        k1dbb = -D2bb*t2bb - Ib.vvoo.copy()
        cc_equations._u_Stanton(k1sa, k1sb, k1daa, k1dab, k1dbb,
                Fa, Fb, Ia, Ib, Iabab, (t1a,t1b), (t2aa,t2ab,t2bb), fac=-1.0)
        return [k1sa,k1sb,k1daa,k1dab,k1dbb]

    for i in range(1,ng):
        # propagate
        h = cc.ti[i] - cc.ti[i - 1]
        d1a,d1b,d2aa,d2ab,d2bb = propagation.rk1(h, [t1a,t1b,t2aa,t2ab,t2bb], fRHS)
        t1a += d1a
        t1b += d1b
        t2aa += d2aa
        t2ab += d2ab
        t2bb += d2bb
        normt = numpy.linalg.norm(uccsd.amplitudes_to_vector((d1a,d1b),(d2aa,d2ab,d2bb)))

        # compute free energy contribution
        dE = g[i]*cc_energy.ucc_energy((t1a,t1b), (t2aa,t2ab,t2bb),
                Fa.ov, Fb.ov, Ia.oovv, Ib.oovv, Iabab.oovv)/beta
        Eccn += dE
        print('cycle = {0}  E(UCCSD) = {1:.10f}  dE = {2:.10f}  norm(t1,t2) = {3:.7}'.format(i, Eccn, dE, normt))

    print('E(UCCSD) = {} E_corr = {}'.format(Eccn+E01, Eccn))
    t1a = t1a.T
    t1b = t1b.T
    t2aa = t2aa.transpose(2,3,0,1)
    t2ab = t2ab.transpose(2,3,0,1)
    t2bb = t2bb.transpose(2,3,0,1)
    return Eccn, (t1a, t1b), (t2aa, t2ab, t2bb)

#def uccsd(cc,T1in=None,T2in=None,denom=None):
#    """Solve finite temperature coupled cluster equations."""
#    # get time-grid
#    ng = cc.ngrid
#    ti = cc.ti
#    G = cc.G
#    g = cc.g
#
#    # get Fock matrix
#    Fa,Fb = cc.sys.u_fock()
#    eoa = numpy.diagonal(Fa.oo)
#    eva = numpy.diagonal(Fa.vv)
#    eob = numpy.diagonal(Fb.oo)
#    evb = numpy.diagonal(Fb.vv)
##    cc.sys.mf.mo_energy = numpy.hstack((eoa,eva)),numpy.hstack((eob,evb))
#
#    # get HF energy
#    En = cc.sys.const_energy()
#    E0 = zt_mp.ump0(eoa, eob) + En
#    E1 = cc.sys.get_mp1()
#    E01 = E0 + E1
#
#    # get ERIs
#    Ia, Ib, Iabab = cc.sys.u_aint_tot()
#
#    # get frozen tensors
#    nmoa, nmob = Iabab.shape[:2]
#    nfca, nfcb = cc.nfc
#    nfva, nfvb = cc.nfv
#    noa, nva, nob, nvb = eoa.size-nfca, eva.size-nfva, eob.size-nfcb, evb.size-nfvb
#    Fa.oo = Fa.oo[nfca:,nfca:] 
#    Fb.oo = Fb.oo[nfcb:,nfcb:]
#    Fa.ov = Fa.ov[nfca:,:nva] 
#    Fb.ov = Fb.ov[nfcb:,:nvb] 
#    Fa.vo = Fa.vo[:nva,nfca:] 
#    Fb.vo = Fb.vo[:nvb,nfcb:] 
#    Fa.vv = Fa.vv[:nva,:nva]
#    Fb.vv = Fb.vv[:nvb,:nvb]
#    if denom is not None:
#        eoa, eob, eva, evb = denom
#    else:
#        eoa, eob, eva, evb = eoa[nfca:],eob[nfcb:],eva[:nva],evb[:nvb] 
#    Fa.oo = Fa.oo - numpy.diag(eoa) # subtract diagonal
#    Fa.vv = Fa.vv - numpy.diag(eva) # subtract diagonal
#    Fb.oo = Fb.oo - numpy.diag(eob) # subtract diagonal
#    Fb.vv = Fb.vv - numpy.diag(evb) # subtract diagonal
#
#    Ia = Ia[nfca:nmoa-nfva,nfca:nmoa-nfva,nfca:nmoa-nfva,nfca:nmoa-nfva]
#    Ib = Ib[nfcb:nmob-nfvb,nfcb:nmob-nfvb,nfcb:nmob-nfvb,nfcb:nmob-nfvb]
#    Iabab = Iabab[nfca:nmoa-nfva,nfcb:nmob-nfvb,nfca:nmoa-nfva,nfcb:nmob-nfvb]
#    Ia = make_two_e_blocks(Ia,noa,nva,noa,nva,noa,nva,noa,nva)
#    Ib = make_two_e_blocks(Ib,nob,nvb,nob,nvb,nob,nvb,nob,nvb)
#    Iabab = make_two_e_blocks_full(Iabab,noa,nva,nob,nvb,noa,nva,nob,nvb)
#    D1a = (eva[:,None] - eoa[None,:])
#    D1b = (evb[:,None] - eob[None,:])
#    D2aa = (eva[:,None,None,None] + eva[None,:,None,None]
#        - eoa[None,None,:,None] - eoa[None,None,None,:])
#    D2ab = (eva[:,None,None,None] + evb[None,:nvb,None,None]
#        - eoa[None,None,:,None] - eob[None,None,None,:])
#    D2bb = (evb[:,None,None,None] + evb[None,:,None,None]
#        - eob[None,None,:,None] - eob[None,None,None,:])
#    print(D1a)
#    print(D1b)
#
#    method = "CCSD" if cc.singles else "CCD"
#    conv_options = {
#            "econv":cc.econv,
#            "tconv":cc.tconv,
#            "max_iter":cc.max_iter,
#            "damp":cc.damp}
#    if cc.rt_iter[0] == 'a' or T2in is not None:
#        if cc.rt_iter[0] != 'a':
#            print("WARNING: Converngece scheme ({}) is being ignored.".format(cc.rt_iter))
#        # get MP2 T-amplitudes
#        if T1in is not None and T2in is not None:
#            T1aold=T1in[0] if cc.singles else numpy.zeros(T1ashape,dtype=Fa.vo.dtype)
#            T1bold=T1in[1] if cc.singles else numpy.zeros(T1bshape,dtype=Fb.vo.dtype)
#            T2aaold = T2in[0]
#            T2abold = T2in[1]
#            T2bbold = T2in[2]
#        else:
#            if cc.singles:
#                Id = numpy.ones((ng))
#                T1aold = -einsum('v,ai->vai',Id,Fa.vo)
#                T1bold = -einsum('v,ai->vai',Id,Fb.vo)
#            else:
#                T1aold = numpy.zeros((ng,nva-nfva,noa-nfca), dtype=Fa.vo.dtype)
#                T1bold = numpy.zeros((ng,nvb-nfvb,nob-nfcb), dtype=Fb.vo.dtype)
#            Id = numpy.ones((ng))
#            T2aaold = -einsum('v,abij->vabij',Id,Ia.vvoo)
#            T2abold = -einsum('v,abij->vabij',Id,Iabab.vvoo)
#            T2bbold = -einsum('v,abij->vabij',Id,Ib.vvoo)
#            T1aold = quadrature.int_tbar1(ng,T1aold,ti,D1a,G)
#            T1bold = quadrature.int_tbar1(ng,T1bold,ti,D1b,G)
#            T2aaold = quadrature.int_tbar2(ng,T2aaold,ti,D2aa,G)
#            T2abold = quadrature.int_tbar2(ng,T2abold,ti,D2ab,G)
#            T2bbold = quadrature.int_tbar2(ng,T2bbold,ti,D2bb,G)
#
#        # MP2 energy
#        E2 = ft_cc_energy.ft_ucc_energy(T1aold,T1bold,T2aaold,T2abold,T2bbold,
#            Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,g,cc.beta_max)
#        if cc.iprint > 0:
#            print('MP2 Energy: {:.10f}'.format(E2))
#
#        # run CC iterations
#        Eccn,T1,T2 = cc_utils.ft_ucc_iter(
#                method, T1aold, T1bold, T2aaold, T2abold, T2bbold,
#                Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
#                g, G, cc.beta_max, ng, ti, cc.iprint, conv_options)
#    else:
#        T1,T2 = cc_utils.ft_ucc_iter_extrap(method, Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
#                g, G, cc.beta_max, ng, ti, cc.iprint, conv_options)
#        Eccn = ft_cc_energy.ft_ucc_energy(T1[0], T1[1], T2[0], T2[1], T2[2],
#            Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,g,cc.beta_max)
#
#    print('E(UCCSD) = {} E_corr = {}'.format(Eccn+E01, Eccn))
#    t1a = einsum('y,yai->ia',g,T1[0])/cc.beta_max 
#    t1b = einsum('y,yai->ia',g,T1[1])/cc.beta_max
#    t2aa = einsum('y,yabij->ijab',g,T2[0])/cc.beta_max 
#    t2ab = einsum('y,yabij->ijab',g,T2[1])/cc.beta_max
#    t2bb = einsum('y,yabij->ijab',g,T2[2])/cc.beta_max
#    return Eccn, (t1a, t1b), (t2aa, t2ab, t2bb)


# doc: this file is not used, nor is mrccsd_.py
# this file containes some functions that might be useful in the future
# mrccsd_.py is the old mrccsd algorithm, basically not useful

from pyscf.fci import cistring
import math
import numpy
from sympy.combinatorics.permutations import Permutation
from itertools import permutations


def gen_occslst(ncore, ncas, nelecas):
    occslst = cistring._gen_occslst(range(ncore, ncore+ncas), nelecas)
    cores = numpy.empty((occslst.shape[0], ncore), dtype=int)
    for i in range(ncore): 
        cores[:, i] = i
    occslst = numpy.hstack((cores, occslst))
    return occslst

def gen_strslst(nc, ncas, necas):
    strs = cistring.make_strings(range(nc, nc+ncas), necas)
    c = 2**nc - 1
    for i in range(len(strs)): 
        strs[i] = strs[i] | c
    return strs 

def str2occ(string, stop):
    orbs = numpy.asarray([1<<p for p in range(stop)])
    hmask = [string & p !=0 for p in orbs]
    pmask = [not p for p in hmask]
    occ = numpy.array(range(stop), dtype=int)[hmask]
    vir = numpy.array(range(stop), dtype=int)[pmask]
    return occ, vir

def det2ind(nmo, target, ref, nfc): 
    ref_ = list(set(range(nmo))-set(ref))
    ref_.sort()
    h_abs = set(ref) - set(target)
    p_abs = set(target) - set(ref)
    h_rel = [ref.index(x) - nfc for x in h_abs]
    p_rel = [ref_.index(x) for x in p_abs]
    return set(p_rel), set(h_rel)

def perm_mo(ncas, necas, mo, occ):
    ne = len(occ)
    nc = ne - necas
    stop = nc + ncas
    mo_ = mo.copy()
    vir = list(set(range(stop))-set(occ)) 
    vir.sort()
    for i in range(nc,ne):
        mo_[:,i] = mo[:,occ[i]].copy()
    for a in range(ne,stop):
        mo_[:,a] = mo[:,vir[a-ne]].copy()
    return mo_

def occ2addr(nmo, nocc, occ):
    strlst = cistring._occslst2strs(numpy.array(occ).reshape(1,len(occ)))  
    return str2addr(nmo, nocc, strlst[0])


def addr2occ(nmo, nocc, addr): 
    string = cistring.addr2str(nmo, nocc, addr)
    occ_arr = cistring._strs2occslst(numpy.array([string]), nmo)
    occ = list(occ_arr.reshape(occ_arr.shape[1]))
    occ.sort()
    return occ

def sign(origin, target, nmo):
    inter = origin.copy()
    sign = 1
    h_abs = set(origin) - set(target)
    p_abs = set(target) - set(origin)
    while len(h_abs) != 0:
        a = p_abs.pop()
        i = h_abs.pop()
        sign *= cistring.cre_des_sign(
                a, i, cistring._occslst2strs(
                numpy.array(inter).reshape(1,len(inter)))[0])
        inter.remove(i)
        inter.append(a)
        inter.sort()
    return sign


# T stored as Ndet^4 tensor
def make_T_(T1, T2, ref, nfc, nfv):
#def make_T(T1, T2, str0, nfc, nfv):
    t1a, t1b = T1
    t2aa, t2ab, t2bb = T2
    noa, nob, nva, nvb = t2ab.shape
    nmoa = nfc[0] + noa + nva + nfv[0]
    nmob = nfc[1] + nob + nvb + nfv[1]
    nea, neb = nfc[0] + noa, nfc[1] + nob
    stpa, stpb = nmoa - nfv[0], nmob - nfv[1]

#    occa, vira = str2occ(str0[0], stpa)
#    occb, virb = str2occ(str0[1], stpb)
    occa, occb = ref 
    vira = list(set(range(stpa))-set(occa))
    virb = list(set(range(stpb))-set(occb))
    vira.sort()
    virb.sort()

    Na = num_strings(nmoa,nea)
    Nb = num_strings(nmob,neb)
    T = numpy.zeros((Na, Nb, Na, Nb))
    temp1 = numpy.zeros((Na, Na, noa, nva))
    temp2 = numpy.zeros((Nb, Nb, nob, nvb))
    for I in range(Na): 
        strI = addr2str(nmoa,nea,I) 
        for i in range(noa):
            des1 = occa[i+nfc[0]]
            h1 = 1 << des1
            if strI & h1 != 0:
                for a in range(nva):
                    cre1 = vira[a]
                    p1 = 1 << cre1
                    if strI & p1 == 0:
                        str1 = strI ^ h1 | p1
                        K = str2addr(nmoa,nea,str1)
                        sgn1 = cre_des_sign(cre1,des1,strI)
                        T[K,range(Nb),I,range(Nb)] += t1a[i,a]*sgn1
                        temp1[K,I,i,a] = sgn1
                        for j in range(i):
                            des2 = occa[j+nfc[0]]
                            h2 = 1 << des2
                            if strI & h2 != 0:
                                for b in range(a):
                                    cre2 = vira[b]
                                    p2 = 1 << cre2
                                    if strI & p2 == 0:
                                        str2 = str1 ^ h2 | p2
                                        K = str2addr(nmoa,nea,str2)
                                        sgn2 = cre_des_sign(cre2,des2,str1)
                                        T[K,range(Nb),I,range(Nb)] += t2aa[i,j,a,b]*sgn1*sgn2
    for I in range(Nb): 
        strI = addr2str(nmob,neb,I) 
        for i in range(nob):
            des1 = occb[i+nfc[1]]
            h1 = 1 << des1
            if strI & h1 != 0:
                for a in range(nvb):
                    cre1 = virb[a]
                    p1 = 1 << cre1
                    if strI & p1 == 0:
                        str1 = strI ^ h1 | p1
                        K = str2addr(nmob,neb,str1)
                        sgn1 = cre_des_sign(cre1,des1,strI)
                        T[range(Na),K,range(Na),I] += t1b[i,a]*sgn1
                        temp2[K,I,i,a] = sgn1
                        for j in range(i):
                            des2 = occb[j+nfc[1]]
                            h2 = 1 << des2
                            if strI & h2 != 0:
                                for b in range(a):
                                    cre2 = virb[b]
                                    p2 = 1 << cre2
                                    if strI & p2 == 0:
                                        str2 = str1 ^ h2 | p2
                                        K = str2addr(nmob,neb,str2)
                                        sgn2 = cre_des_sign(cre2,des2,str1)
                                        T[range(Na),K,range(Na),I] += t2bb[i,j,a,b]*sgn1*sgn2
    T += einsum('ijab,KIia,LJjb->KLIJ',t2ab,temp1,temp2)
    return T

def make_ci(T1, T2, P, nmo, nele):
    # version 1, requires explicit t>ci equations
    Pa, Pb = P
#    strPa = cistring._occslst2strs(np.array(Pa).reshape(1,len(Pa)))[0]  
#    strPb = cistring._occslst2strs(np.array(Pb).reshape(1,len(Pb)))[0]  
    Ndet_a = cistring.num_strings(nmo[0], nele[0])
    Ndet_b = cistring.num_strings(nmo[1], nele[1])
    civec = np.zeros((Ndet_a, Ndet_b))
    for i in range(Ndet_a):
        for j in range(Ndet_b):
#            str_a = cistring.addr2str(nmo[0], nele[0], i) 
#            str_b = cistring.addr2str(nmo[0], nele[0], j)
            occ_a = mrccsd_utils.addr2occ(nmo[0], nele[0], i)
            occ_b = mrccsd_utils.addr2occ(nmo[1], nele[1], j)
            sign_a = mrccsd_utils.sign(Pa, occ_a, nmo[0])
            sign_b = mrccsd_utils.sign(Pb, occ_b, nmo[1])
            pa, ha = mrccsd_utils.det2ind(nmo[0], occ_a, Pa)
            pb, hb = mrccsd_utils.det2ind(nmo[1], occ_b, Pb)
            if len(pa+pb+ha+hb) == 0: # diagonal
                civec[i,j] += 1.0
            if len(pa+pb+ha+hb) == 2: # singles
                civec[i,j] += s1(T1, pa, ha, pb, hb)*sign_a*sign_b
            if len(pa+pb+ha+hb) == 4: # doubles
                civec[i,j] += s2(T1, T2, pa, ha, pb, hb)*sign_a*sign_b
            if len(pa+pb+ha+hb) == 6: # triples
                civec[i,j] += s3(T1, T2, pa, ha, pb, hb)*sign_a*sign_b
    return civec


def ovlp3(t1, t2, i,j,k,a,b,c):
#    perm_o = [[i,j,k,1],[j,i,k,-1],[k,j,i,-1]]
#    perm_v = [[a,b,c,1],[b,a,c,-1],[c,b,a,-1]]
#    c3_ = 0.0
#    for i_,j_,k_,sign_o in perm_o:
#        for a_,b_,c_,sign_v in perm_v: 
#            c3_ += t1[i_,a_]*t2[j_,k_,b_,c_]*sign_o*sign_v
#    for [i_,j_,k_],sign in perm([i,j,k]):
#        c3_ += t1[i_,a]*t1[j_,b]*t1[k_,c]

    c3 = t1[i,a]*t2[j,k,b,c]-t1[i,b]*t2[j,k,a,c]-t1[i,c]*t2[j,k,b,a]
    c3 += -t1[j,a]*t2[i,k,b,c]+t1[j,b]*t2[i,k,a,c]-t1[j,c]*t2[i,k,a,b]
    c3 += -t1[k,a]*t2[j,i,b,c]-t1[k,b]*t2[i,j,a,c]+t1[k,c]*t2[i,j,a,b]
    c3 += t1[i,a]*t1[j,b]*t1[k,c]+t1[j,a]*t1[k,b]*t1[i,c]+t1[k,a]*t1[i,b]*t1[j,c]
    c3 += -t1[i,a]*t1[k,b]*t1[j,c]-t1[j,a]*t1[i,b]*t1[k,c]-t1[k,a]*t1[j,b]*t1[i,c]
    if abs(c3-c3_)<1e-8:
        print('error: ', c3-c3_)
    return c3

def ovlp3_(t1, t1_, t2, t2_, i,j,k,a,b,c): # assumes k,c of different spin
#    perm_o = [[i,j,k,1],[j,i,k,-1]]
#    perm_v = [[a,b,c,1],[b,a,c,-1]]
#    c3_ = 0.0
#    for i_,j_,k_,sign_o in perm_o:
#        for a_,b_,c_,sign_v in perm_v: 
#            c3_ += t1[i_,a_]*t2[j_,k_,b_,c_]*sign_o*sign_v
#    for [i_,j_,k_],sign in perm([i,j]):
#        c3_ += t1[i_,a]*t1[j_,b]*t1[k_,c]
#    c3_ += t1_[k,c]*t2[i,j,a,b]

    c3 = t1[i,a]*t2_[j,k,b,c]-t1[i,b]*t2_[j,k,a,c]
    c3 += -t1[j,a]*t2_[i,k,b,c]+t1[j,b]*t2_[i,k,a,c]
    c3 += t1_[k,c]*t2[i,j,a,b]
    c3 += t1[i,a]*t1[j,b]*t1_[k,c]-t1[j,a]*t1[i,b]*t1_[k,c]
    if abs(c3-c3_)<1e-8:
        print('error: ', c3-c3_)
    return c3



def ovlp3(t1, t2, i,j,k,a,b,c):
    c3 = t1[i,a]*t2[j,k,b,c]-t1[i,b]*t2[j,k,a,c]-t1[i,c]*t2[j,k,b,a]
    c3 += -t1[j,a]*t2[i,k,b,c]+t1[j,b]*t2[i,k,a,c]-t1[j,c]*t2[i,k,a,b]
    c3 += -t1[k,a]*t2[j,i,b,c]-t1[b,k]*t2[i,j,a,c]+t1[k,c]*t2[i,j,a,b]
    c3 += t1[i,a]*t1[j,b]*t1[k,c]+t1[j,a]*t1[k,b]*t1[i,c]+t1[k,a]*t1[i,b]*t1[j,c]
    c3 += -t1[i,a]*t1[k,b]*t1[j,c]-t1[j,a]*t1[i,b]*t1[k,c]-t1[k,a]*t1[j,b]*t1[i,c]
    return c3

def ovlp3_(t1, t1_, t2, t2_, i,j,k,a,b,c):
    c3 = t1[i,a]*t2_[j,k,b,c]-t1[i,b]*t2_[j,k,a,c]
    c3 += -t1[j,a]*t2_[i,k,b,c]+t1[j,b]*t2_[i,k,a,c]
    c3 += t1_[k,c]*t2[i,j,a,b]
    c3 += t1[i,a]*t1[j,b]*t1_[k,c]-t1[j,a]*t1[i,b]*t1_[k,c]
    return c3

def regularizer(reg):
    reg, param, p = reg
    def alpha(e):
        return 1./np.power(-param+e,p) + e
    def sigma(e):
        fac = 1. - np.exp(-param*np.power(-e,p))
        return np.divide(e,fac)
    def kappa(e):
        fac = 1. - np.exp(-param*np.power(-e,p))
        return np.divide(e,np.power(fac,2))
    if reg[0] == 'k':
        return kappa
    if reg[0] == 's':
        return sigma
    if reg[0] == 'a':
        return alpha
#    if lumo is not None: 
#        # setting hole orbital energies to LUMO
#        vira0, virb0 = set(range(nea,nmoa)), set(range(neb,nmob))
#        vira, virb = list(vir[:nmoa-nea]), list(vir[nmoa-nea:])
#        idxa = [vira.index(x) for x in set(vira) - vira0]
#        idxb = [virb.index(x) for x in set(virb) - virb0]
#        eva[idxa] = lumo[0]
#        evb[idxb] = lumo[1]
#    eia_a = lib.direct_sum('i-a->ia', eoa, eva)
#    eia_b = lib.direct_sum('i-a->ia', eob, evb)
#    eijab_aa = lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
#    eijab_ab = lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
#    eijab_bb = lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
#    if reg is not None:
#        reg = regularizer(reg)
#        eia_a = reg(eia_a)
#        eia_b = reg(eia_b)
#        eijab_aa = reg(eijab_aa)
#        eijab_ab = reg(eijab_ab)
#        eijab_bb = reg(eijab_bb)
#    return (eia_a,eia_b,eijab_aa,eijab_ab,eijab_bb)

def update_amps(cc, t1, t2, eris, e1, e2, method):
    if method == 'ccsd':
        t1new, t2new = uccsd.update_amps(cc, t1, t2, eris)
    if method == 'cisd':
        t1, t2 = make_ci(t1, t2)
        t1new, t2new = update_ci(cc, t1, t2, eris)
        e0 = cc._scf.e_tot
        t1a = np.multiply(t1[0], e0*np.ones_like(e1[0])/e1[0]+1.0)
        t1b = np.multiply(t1[1], e0*np.ones_like(e1[1])/e1[1]+1.0)
        t2aa = np.multiply(t2[0], e0*np.ones_like(e2[0])/e2[0]+1.0)
        t2ab = np.multiply(t2[1], e0*np.ones_like(e2[1])/e2[1]+1.0)
        t2bb = np.multiply(t2[2], e0*np.ones_like(e2[2])/e2[2]+1.0)
        t1, t2 = (t1a, t1b), (t2aa, t2ab, t2bb)

    dt1a = np.multiply(e1[0], t1new[0]-t1[0])
    dt1b = np.multiply(e1[1], t1new[1]-t1[1])
    dt2aa = np.multiply(e2[0], t2new[0]-t2[0])
    dt2ab = np.multiply(e2[1], t2new[1]-t2[1])
    dt2bb = np.multiply(e2[2], t2new[2]-t2[2])
    return (dt1a, dt1b), (dt2aa, dt2ab, dt2bb)

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

def update_ci(cc, t1, t2, eris):
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
    u1a += einsum('ic,ac->ia',t1a,Fvva)
    u1a -= einsum('ik,ka->ia',Fooa,t1a)
    u1a += einsum('kcai,kc->ia',eris.ovvo,t1a)
    u1a -= einsum('kiac,kc->ia',eris.oovv,t1a)
    u1a += einsum('KCai,KC->ia',eris.OVvo,t1b)
    u1a += einsum('kc,ikac->ia',fova,t2aa)
    u1a += einsum('KC,iKaC->ia',fovb,t2ab)
#    u1a += 0.5*einsum('kcad,kicd->ia',eris.ovvv,t2aa)
#    u1a -= 0.5*einsum('kdac,kicd->ia',eris.ovvv,t2aa)
#    u1a += einsum('KCad,iKdC->ia',eris.OVvv,t2ab)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now - t2aa.size*8e-6)
    if nvira > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            u1a += 0.5*lib.einsum('kicd,kcad->ia', t2aa[p0:p1], ovvv)
            ovvv = None
    if nvira > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            u1a += lib.einsum('iKdC,KCad->ia', t2ab[:,p0:p1], OVvv)
            OVvv = None
    u1a -= 0.5*einsum('kcli,klca->ia',eris.ovoo,t2aa)
    u1a += 0.5*einsum('lcki,klca->ia',eris.ovoo,t2aa)
    u1a -= einsum('KCli,lKaC->ia',eris.OVoo,t2ab)

    u1b = fovb.copy()
    u1b += einsum('IC,AC->IA',t1b,Fvvb)
    u1b -= einsum('IK,KA->IA',Foob,t1b)
    u1b += einsum('KCAI,KC->IA',eris.OVVO,t1b)
    u1b -= einsum('KIAC,KC->IA',eris.OOVV,t1b)
    u1b += einsum('kcAI,kc->IA',eris.ovVO,t1a)
    u1b += einsum('KC,IKAC->IA',fovb,t2bb)
    u1b += einsum('kc,kIcA->IA',fova,t2ab)
#    u1b += 0.5*einsum('KCAD,KICD->IA',eris.OVVV,t2bb)
#    u1b -= 0.5*einsum('KDAC,KICD->IA',eris.OVVV,t2bb)
#    u1b += einsum('kcAD,kIcD->IA',eris.ovVV,t2ab)
    if nvirb > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            u1b += 0.5*lib.einsum('KICD,KCAD->IA', t2bb[p0:p1], OVVV)
            OVVV = None
    if nvirb > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            u1b += lib.einsum('kIcD,kcAD->IA', t2ab[p0:p1], ovVV)
            ovVV = None
    u1b -= 0.5*einsum('KCLI,KLCA->IA',eris.OVOO,t2bb)
    u1b += 0.5*einsum('LCKI,KLCA->IA',eris.OVOO,t2bb)
    u1b -= einsum('kcLI,kLcA->IA',eris.ovOO,t2ab)

    u2aa, u2ab, u2bb = cc._add_vvvv(None, t2, eris)
    u2aa *= 0.5
    u2bb *= 0.5
    u2aa -= u2aa.transpose(0,1,3,2)
    u2bb -= u2bb.transpose(0,1,3,2)

    u2aa += eris.ovov.transpose(0,2,1,3) - eris.ovov.transpose(0,2,3,1) 
    u2aa += einsum('bc,ijac->ijab',Fvva,t2aa)
    u2aa += einsum('ac,ijcb->ijab',Fvva,t2aa)
    u2aa -= einsum('kj,ikab->ijab',Fooa,t2aa)
    u2aa -= einsum('ki,kjab->ijab',Fooa,t2aa)
    u2aa += 0.5*einsum('kilj,klab->ijab',eris.oooo,t2aa)
    u2aa -= 0.5*einsum('kjli,klab->ijab',eris.oooo,t2aa)
#    u2aa += 0.5*einsum('acbd,ijcd->ijab',eris.vvvv,t2aa)
#    u2aa -= 0.5*einsum('adbc,ijcd->ijab',eris.vvvv,t2aa)
    temp  = einsum('kcbj,ikac->ijab',eris.ovvo,t2aa)
    temp -= einsum('kjbc,ikac->ijab',eris.oovv,t2aa)
    temp += einsum('KCbj,iKaC->ijab',eris.OVvo,t2ab)
    temp -= temp.transpose(1,0,2,3)
    temp -= temp.transpose(0,1,3,2)
    u2aa += temp
    if nvira > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            temp  = einsum('jbca,ic->ijab',ovvv,t1a)
            temp -= einsum('jacb,ic->ijab',ovvv,t1a) 
            temp -= temp.transpose(1,0,2,3)
            u2aa[:,p0:p1] += temp
            ovvv = temp = None
    temp  = einsum('jbik,ka->ijab',eris.ovoo,t1a)
    temp -= einsum('ibjk,ka->ijab',eris.ovoo,t1a)
    temp -= temp.transpose(0,1,3,2)
    u2aa -= temp
    # disconnected term
    temp = einsum('ia,jb->ijab',fova,t1a)
    temp -= temp.transpose(1,0,2,3)
    temp -= temp.transpose(0,1,3,2)
    u2aa += temp

    u2bb += eris.OVOV.transpose(0,2,1,3) - eris.OVOV.transpose(0,2,3,1) 
    u2bb += einsum('BC,IJAC->IJAB',Fvvb,t2bb)
    u2bb += einsum('AC,IJCB->IJAB',Fvvb,t2bb)
    u2bb -= einsum('KJ,IKAB->IJAB',Foob,t2bb)
    u2bb -= einsum('KI,KJAB->IJAB',Foob,t2bb)
    u2bb += 0.5*einsum('KILJ,KLAB->IJAB',eris.OOOO,t2bb)
    u2bb -= 0.5*einsum('KJLI,KLAB->IJAB',eris.OOOO,t2bb)
#    u2bb += 0.5*einsum('ACBD,IJCD->IJAB',eris.VVVV,t2bb)
#    u2bb -= 0.5*einsum('ADBC,IJCD->IJAB',eris.VVVV,t2bb)
    temp  = einsum('KCBJ,IKAC->IJAB',eris.OVVO,t2bb)
    temp -= einsum('KJBC,IKAC->IJAB',eris.OOVV,t2bb)
    temp += einsum('kcBJ,kIcA->IJAB',eris.ovVO,t2ab)
    temp -= temp.transpose(1,0,2,3)
    temp -= temp.transpose(0,1,3,2)
    u2bb += temp
    if nvirb > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            temp  = einsum('JBCA,IC->IJAB',OVVV,t1b)
            temp -= einsum('JACB,IC->IJAB',OVVV,t1b) 
            temp -= temp.transpose(1,0,2,3)
            u2bb[:,p0:p1] += temp
            OVVV = temp = None
    temp  = einsum('JBIK,KA->IJAB',eris.OVOO,t1b)
    temp -= einsum('IBJK,KA->IJAB',eris.OVOO,t1b)
    temp -= temp.transpose(0,1,3,2)
    u2bb -= temp
    # disconnected term
    temp = einsum('IA,JB->IJAB',fovb,t1b)
    temp -= temp.transpose(1,0,2,3)
    temp -= temp.transpose(0,1,3,2)
    u2bb += temp

    u2ab += eris.ovOV.transpose(0,2,1,3)
    u2ab += einsum('BC,iJaC->iJaB',Fvvb,t2ab)
    u2ab += einsum('ac,iJcB->iJaB',Fvva,t2ab)
    u2ab -= einsum('KJ,iKaB->iJaB',Foob,t2ab)
    u2ab -= einsum('ki,kJaB->iJaB',Fooa,t2ab)
    u2ab += einsum('kiLJ,kLaB->iJaB',eris.ooOO,t2ab)
#    u2ab += einsum('acBD,iJcD->iJaB',eris.vvVV,t2ab)
    u2ab += einsum('kcBJ,ikac->iJaB',eris.ovVO,t2aa)
    u2ab += einsum('KCai,JKBC->iJaB',eris.OVvo,t2bb)
    u2ab += einsum('KCBJ,iKaC->iJaB',eris.OVVO,t2ab)
    u2ab -= einsum('KJBC,iKaC->iJaB',eris.OOVV,t2ab)
    u2ab -= einsum('kiBC,kJaC->iJaB',eris.ooVV,t2ab)
    u2ab -= einsum('KJac,iKcB->iJaB',eris.OOvv,t2ab)
    u2ab += einsum('kcai,kJcB->iJaB',eris.ovvo,t2ab)
    u2ab -= einsum('kiac,kJcB->iJaB',eris.oovv,t2ab)
    if nvira > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            u2ab[p0:p1] += einsum('JBca,ic->iJaB',OVvv,t1a)
            OVvv = None
    if nvirb > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            u2ab[p0:p1] += einsum('iaCB,JC->iJaB',ovVV,t1b)
            ovVV = tmp1ab = None
    u2ab -= einsum('JBik,ka->iJaB',eris.OVoo,t1a)
    u2ab -= einsum('iaJK,KB->iJaB',eris.ovOO,t1b)
    # disconnected term
    u2ab += einsum('ia,JB->iJaB',fova,t1b)
    u2ab += einsum('JB,ia->iJaB',fovb,t1a)

#    e = energy_ci(t1, t2, eris)
#    u1a -= e * t1a
#    u1b -= e * t1b
#    u2aa -= e * t2aa
#    u2ab -= e * t2ab
#    u2bb -= e * t2bb

#    time0 = log.timer_debug1('update t1 t2', *time0)
#    return (-u1a, -u1b), (-u2aa, -u2ab, -u2bb)
    return (u1a, u1b), (u2aa, u2ab, u2bb)


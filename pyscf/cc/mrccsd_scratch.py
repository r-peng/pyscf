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


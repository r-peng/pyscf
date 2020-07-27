import numpy as np
from pyscf import lib, ao2mo
einsum = lib.einsum

def sort1(tup):
    a, b = tup
    na0, na1 = a.shape
    nb0, nb1 = b.shape
    out = np.zeros((na0+nb0,na1+nb1))
    out[ ::2, ::2] = a.copy()
    out[1::2,1::2] = b.copy()
    return out

def sort2(tup, anti):
    aa, ab, bb = tup
    na0, na1, na2, na3 = aa.shape
    nb0, nb1, nb2, nb3 = bb.shape
    out = np.zeros((na0+nb0,na1+nb1,na2+nb2,na3+nb3))
    out[ ::2, ::2, ::2, ::2] = aa.copy() 
    out[1::2,1::2,1::2,1::2] = bb.copy() 
    out[ ::2,1::2, ::2,1::2] = ab.copy()
    out[1::2, ::2,1::2, ::2] = ab.transpose(1,0,3,2).copy()
    if anti:
        out[ ::2,1::2,1::2, ::2] = - ab.transpose(0,1,3,2).copy()
        out[1::2, ::2, ::2,1::2] = - ab.transpose(1,0,2,3).copy()
    return out

def update_T(t, eris):
    Foo  = eris.foo.copy()
    Foo += 0.5 * einsum('klcd,cdjl->kj',eris.oovv,t)
    Fvv  = eris.fvv.copy()
    Fvv -= 0.5 * einsum('klcd,bdkl->bc',eris.oovv,t)

    dt  = eris.oovv.transpose(2,3,0,1).conj().copy()
    dt += einsum('bc,acij->abij',Fvv,t)
    dt += einsum('ac,cbij->abij',Fvv,t)
    dt -= einsum('kj,abik->abij',Foo,t)
    dt -= einsum('ki,abkj->abij',Foo,t)

    dt += 0.5 * einsum('klij,abkl->abij',eris.oooo,t)
    dt += 0.5 * einsum('abcd,cdij->abij',eris.vvvv,t)
    tmp = 0.5 * einsum('klcd,cdij->klij',eris.oovv,t)
    dt += 0.5 * einsum('klij,abkl->abij',tmp,t)

    tmp  = eris.ovvo.copy()
    tmp += 0.5 * einsum('klcd,bdjl->kbcj',eris.oovv,t)
    tmp  = einsum('kbcj,acik->abij',tmp,t)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dt += tmp.copy()
    return dt

def update_L(t, l, eris):
    Foo  = eris.foo.copy()
    Foo += 0.5 * einsum('ilcd,cdkl->ik',eris.oovv,t)
    Fvv  = eris.fvv.copy()
    Fvv -= 0.5 * einsum('klad,cdkl->ca',eris.oovv,t)
    
    dl  = eris.oovv.copy() 
    dl += einsum('ca,ijcb->ijab',Fvv,l)
    dl += einsum('cb,ijac->ijab',Fvv,l)
    dl -= einsum('ik,kjab->ijab',Foo,l)
    dl -= einsum('jk,ikab->ijab',Foo,l)

    tmp  = 0.5 * einsum('ilcd,cdkl->ik',l,t)
    dl -= einsum('ik,kjab->ijab',tmp,eris.oovv)
    dl -= einsum('jk,ikab->ijab',tmp,eris.oovv)
    tmp  = 0.5 * einsum('klad,cdkl->ca',l,t)
    dl -= einsum('ca,ijcb->ijab',tmp,eris.oovv)
    dl -= einsum('cb,ijac->ijab',tmp,eris.oovv)

    vvvv  = eris.vvvv.copy()
    vvvv += 0.5 * einsum('klab,cdkl->cdab',eris.oovv,t)
    oooo  = eris.oooo.copy()
    oooo += 0.5 * einsum('ijcd,cdkl->ijkl',eris.oovv,t)

    dl += 0.5 * einsum('cdab,ijcd->ijab',vvvv,l)
    dl += 0.5 * einsum('ijkl,klab->ijab',oooo,l)

    ovvo  = eris.ovvo.copy()
    ovvo += einsum('jlbd,cdkl->jcbk',eris.oovv,t)
    tmp  = einsum('jcbk,ikac->ijab',ovvo,l)
    tmp += tmp.transpose(1,0,3,2)
    tmp -= tmp.transpose(0,1,3,2)
    dl += tmp.copy()
    return dl

def gamma1(t, l): # normal ordered
    dvv = 0.5 * einsum('ikac,bcik->ba',l,t)
    doo = - 0.5 * einsum('jkac,acik->ji',l,t)
    return doo, dvv

def gamma2(t, l):
    doovv = l.copy()
    dovvo = einsum('jkbc,acik->jabi',l,t)
    dvvvv = 0.5 * einsum('ijab,cdij->cdab',l,t)
    doooo = 0.5 * einsum('klab,abij->klij',l,t)
    dvvoo = t.copy()
    tmp  = einsum('acik,klcd->alid',t,l)
    tmp  = einsum('alid,bdjl->abij',tmp,t)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo += tmp.copy()
    tmp  = einsum('adkl,klcd->ac',t,l)
    tmp  = einsum('ac,cbij->abij',tmp,t)
    tmp -= tmp.transpose(1,0,2,3)
    dvvoo -= 0.5 * tmp.copy()
    tmp  = einsum('cdil,klcd->ki',t,l)
    tmp  = einsum('ki,abkj->abij',tmp,t)
    tmp -= tmp.transpose(0,1,3,2)
    dvvoo -= 0.5 * tmp.copy()
    tmp  = einsum('cdij,klcd->klij',t,l)
    dvvoo += 0.25 * einsum('klij,abkl->abij',tmp,t)
    return doooo, doovv, dvvoo, dovvo, dvvvv

def rdm(t, l): # asymmetric
    doo, dvv = gamma1(t, l)
    doooo, doovv, dvvoo, dovvo, dvvvv = gamma2(t, l)

    no, nv = doo.shape[0], dvv.shape[0]
    nmo = no + nv
    doooo += einsum('ki,lj->klij',np.eye(no),doo)
    doooo += einsum('lj,ki->klij',np.eye(no),doo)
    doooo -= einsum('li,kj->klij',np.eye(no),doo)
    doooo -= einsum('kj,li->klij',np.eye(no),doo)
    doooo += einsum('ki,lj->klij',np.eye(no),np.eye(no))
    doooo -= einsum('li,kj->klij',np.eye(no),np.eye(no))
    dovvo -= einsum('ji,ab->jabi',np.eye(no),dvv)
    doo += np.eye(no)

    d1 = np.zeros((nmo,nmo))
    d1[:no,:no] = doo.copy()
    d1[no:,no:] = dvv.copy()
    d2 = np.zeros((nmo,nmo,nmo,nmo))
    d2[:no,:no,:no,:no] = doooo.copy()
    d2[:no,:no,no:,no:] = doovv.copy()
    d2[no:,no:,:no,:no] = dvvoo.copy()
    d2[:no,no:,no:,:no] = dovvo.copy()
    d2[no:,:no,:no,no:] = dovvo.transpose(1,0,3,2)
    d2[:no,no:,:no,no:] = - dovvo.transpose(0,1,3,2)
    d2[no:,:no,no:,:no] = - dovvo.transpose(1,0,2,3)
    d2[no:,no:,no:,no:] = dvvvv.copy()

    d1 = 0.5 * (d1 + d1.T.conj())
    d2 = 0.5 * (d2 + d2.transpose(2,3,0,1).conj())
    return d1, d2

class ERIs:
    def __init__(self, mf):
        nmo = mf.mol.nao_nr()
        noa, nob = mf.mol.nelec
        no = noa + nob

        f0 = np.diag(mf.mo_energy)
        f0 = sort1((f0, f0)).astype(complex)
        self.foo = f0[:no,:no].copy()
        self.fov = f0[:no,no:].copy()
        self.fvv = f0[no:,no:].copy()

        eri = mf.mol.intor('int2e_sph', aosym='s8')
        eri = ao2mo.incore.full(eri, mf.mo_coeff)
        eri = ao2mo.restore(1, eri, nmo)
        eri = eri.transpose(0,2,1,3)
        eri = sort2((eri, eri, eri), anti=False).astype(complex)
        eri -= eri.transpose(0,1,3,2)
        self.oovv = eri[:no,:no,no:,no:].copy() 
        self.ovvo = eri[:no,no:,no:,:no].copy() 
        self.oooo = eri[:no,:no,:no,:no].copy() 
        self.vvvv = eri[no:,no:,no:,no:].copy() 

 

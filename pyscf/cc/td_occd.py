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

    dt  = eris.oovv.transpose(2,3,0,1).conj()
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

class ERIs:
    def __init__(self, mf, mo_coeff=None):
        mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        nmo = mf.mol.nao_nr()
        noa, nob = mf.mol.nelec
        no = noa + nob

        f0 = np.diag(mf.mo_energy)
        f0 = sort1((f0, f0)).astype(complex)
        self.foo = f0[:no,:no].copy()
        self.fov = f0[:no,no:].copy()
        self.fvv = f0[no:,no:].copy()

        eri = mf.mol.intor('int2e_sph', aosym='s8')
        eri = ao2mo.incore.full(eri, mo_coeff)
        eri = ao2mo.restore(1, eri, nmo)
        eri = eri.transpose(0,2,1,3)
        eri = sort2((eri, eri, eri), anti=False).astype(complex)
        eri -= eri.transpose(0,1,3,2)
#        print('interal symmetry: {}'.format(np.linalg.norm(eri+eri.transpose(1,0,2,3)))) 
#        print('interal symmetry: {}'.format(np.linalg.norm(eri+eri.transpose(0,1,3,2)))) 
#        print('interal symmetry: {}'.format(np.linalg.norm(eri-eri.transpose(1,0,3,2))))

        self.oovv = eri[:no,:no,no:,no:].copy() 
        self.ovvo = eri[:no,no:,no:,:no].copy() 
        self.oooo = eri[:no,:no,:no,:no].copy() 
        self.vvvv = eri[no:,no:,no:,no:].copy() 
 

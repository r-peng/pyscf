import numpy as np
import scipy
from pyscf import lib, cc
from pyscf.cc import td_uoccd_utils as utils
einsum = lib.einsum

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

def kernel(eris, t, l, tf, step, RK=4):
    nmo = eris.mf.mol.nao_nr()
    N = int((tf+step*0.1)/step)
    Ca, Cb = np.eye(nmo, dtype=complex), np.eye(nmo, dtype=complex)

    taa = np.array(t[0], dtype=complex)
    tab = np.array(t[1], dtype=complex)
    tbb = np.array(t[2], dtype=complex)
    laa = np.array(l[0], dtype=complex)
    lab = np.array(l[1], dtype=complex)
    lbb = np.array(l[2], dtype=complex)
    t, l = (taa, tab, tbb), (laa, lab, lbb)
    d1, d2 = utils.compute_rdm12(t, l)
    e = utils.compute_energy(d1, d2, eris, time=None)
    print('check initial energy: {}'.format(e.real+eris.mf.energy_nuc())) 

    d1a_old, d1b_old = build_rdm1(d1)
    E = np.zeros(N+1,dtype=complex) 
#    mu = np.zeros((N+1,3),dtype=complex)  
    for i in range(N+1):
        time = i * step
        dt, dl, X, E[i], F = utils.update_RK(t, l, (Ca,Cb), eris, time, step, RK)
#        mu[i,:] = einsum('qp,xpq->x',utils.rotate1(d1,C.T.conj()),eris.mu_) 
        # update 
        taa = t[0] + step * dt[0]
        tab = t[1] + step * dt[1]
        tbb = t[2] + step * dt[2]
        laa = l[0] + step * dl[0]
        lab = l[1] + step * dl[1]
        lbb = l[2] + step * dl[2]
        t, l = (taa, tab, tbb), (laa, lab, lbb)
        Ca = np.dot(scipy.linalg.expm(-step*X[0]), Ca)
        Cb = np.dot(scipy.linalg.expm(-step*X[1]), Cb)
        d1 = utils.compute_rdm1(t, l)
        d1a_new, d1b_new = build_rdm1(d1) 
        d1a_new = utils.rotate1(d1a_new, Ca.T.conj())
        d1b_new = utils.rotate1(d1b_new, Ca.T.conj())
        if RK == 1:
            # Ehrenfest error
            err  = np.linalg.norm((d1a_new-d1a_old)/step-1j*F[0])
            err += np.linalg.norm((d1b_new-d1b_old)/step-1j*F[1])
            print('time: {:.4f}, EE(mH): {}, Xa: {}, Xb: {}, err: {}'.format(
                  time, (E[i] - E[0]).real*1e3, 
                  np.linalg.norm(X[0]), np.linalg.norm(X[1]), err))
        else:
            print('time: {:.4f}, EE(mH): {}, Xa: {}, Xb: {}'.format(
                  time, (E[i] - E[0]).real*1e3, 
                  np.linalg.norm(X[0]), np.linalg.norm(X[1])))
        d1a_old = d1a_new.copy()
        d1b_old = d1b_new.copy()
    return (d1a_old,d1b_old), F, (Ca,Cb), X, t, l

class ERIs_mol:
    def __init__(self, mf, z=np.zeros(3), w=0.0, td=0.0):
        self.mf = mf
        self.w = w
        self.td = td
        self.h0_, self.h1_, self.eri_ = utils.mo_ints_mol(mf, z)[:3]
        self.picture = 'S'

        h0a = np.array(self.h0_[0],dtype=complex)
        h0b = np.array(self.h0_[1],dtype=complex)
        h1a = np.array(self.h1_[0],dtype=complex)
        h1b = np.array(self.h1_[1],dtype=complex)
        eriaa = np.array(self.eri_[0],dtype=complex)
        eriab = np.array(self.eri_[1],dtype=complex)
        eribb = np.array(self.eri_[2],dtype=complex)
        self.h0 = h0a, h0b
        self.h1 = h1a, h1b
        self.eri = eriaa, eriab, eribb

    def rotate(self, Ca, Cb):
        h0a = utils.rotate1(self.h0_[0], Ca)
        h0b = utils.rotate1(self.h0_[1], Cb)
        h1a = utils.rotate1(self.h1_[0], Ca)
        h1b = utils.rotate1(self.h1_[1], Cb)
        eriaa = utils.rotate2(self.eri_[0], Ca, Ca)
        eriab = utils.rotate2(self.eri_[1], Ca, Cb)
        eribb = utils.rotate2(self.eri_[2], Cb, Cb)
        self.h0 = h0a, h0b
        self.h1 = h1a, h1b
        self.eri = eriaa, eriab, eribb

    def make_tensors(self, time=None):
        noa, nob = self.mf.mol.nelec
        ha, hb = self.h0[0].copy(), self.h0[1].copy()
        if time is not None:
            fac = utils.fac_mol(self.w, self.td, time)
            ha += self.h1[0] * fac
            hb += self.h1[1] * fac

        self.hoo = ha[:noa,:noa].copy()
        self.hvv = ha[noa:,noa:].copy()
        self.hov = ha[:noa,noa:].copy()
        self.oovv = self.eri[0][:noa,:noa,noa:,noa:].copy()
        self.oooo = self.eri[0][:noa,:noa,:noa,:noa].copy()
        self.vvvv = self.eri[0][noa:,noa:,noa:,noa:].copy()
        self.ovvo = self.eri[0][:noa,noa:,noa:,:noa].copy()
        self.ovvv = self.eri[0][:noa,noa:,noa:,noa:].copy()
        self.ooov = self.eri[0][:noa,:noa,:noa,noa:].copy()

        self.hOO = hb[:nob,:nob].copy()
        self.hVV = hb[nob:,nob:].copy()
        self.hOV = hb[:nob,nob:].copy()
        self.OOVV = self.eri[2][:nob,:nob,nob:,nob:].copy()
        self.OOOO = self.eri[2][:nob,:nob,:nob,:nob].copy()
        self.VVVV = self.eri[2][nob:,nob:,nob:,nob:].copy()
        self.OVVO = self.eri[2][:nob,nob:,nob:,:nob].copy()
        self.OVVV = self.eri[2][:nob,nob:,nob:,nob:].copy()
        self.OOOV = self.eri[2][:nob,:nob,:nob,nob:].copy()

        self.oOvV = self.eri[1][:noa,:nob,noa:,nob:].copy()
        self.oOoO = self.eri[1][:noa,:nob,:noa,:nob].copy()
        self.vVvV = self.eri[1][noa:,nob:,noa:,nob:].copy()
        self.oVvO = self.eri[1][:noa,nob:,noa:,:nob].copy()
        self.oVoV = self.eri[1][:noa,nob:,:noa,nob:].copy()
        self.vOvO = self.eri[1][noa:,:nob,noa:,:nob].copy()
        self.oVvV = self.eri[1][:noa,nob:,noa:,nob:].copy()
        self.vOvV = self.eri[1][noa:,:nob,noa:,nob:].copy()
        self.oOoV = self.eri[1][:noa,:nob,:noa,nob:].copy()
        self.oOvO = self.eri[1][:noa,:nob,noa:,:nob].copy()

        self.foo, self.fvv = self.hoo.copy(), self.hvv.copy()
        self.fOO, self.fVV = self.hOO.copy(), self.hVV.copy()
        self.foo += einsum('piqi->pq',self.oooo)
        self.foo += einsum('pIqI->pq',self.oOoO)
        self.fOO += einsum('piqi->pq',self.OOOO)
        self.fOO += einsum('IpIq->pq',self.oOoO)
        self.fvv -= einsum('ipqi->pq',self.ovvo)
        self.fvv += einsum('pIqI->pq',self.vOvO)
        self.fVV -= einsum('ipqi->pq',self.OVVO)
        self.fVV += einsum('IpIq->pq',self.oVoV)


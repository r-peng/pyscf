import numpy as np
from pyscf import lib, cc
from pyscf.cc import td_uoccd_utils as utils
import scipy
einsum = lib.einsum

def _U(fd, fd_):
    return np.block([[np.diag(fd),np.diag(fd_)],
                     [np.diag(fd_),-np.diag(fd)]])

def kernel(eris, t, l, tf, step, RK=4, every=1):
    (fda, fda_), (fdb, fdb_) = eris.fd
    ua, ub = eris.mo_coeff
    nmo = len(fda)
    N = int((tf+step*0.1)/step)
    Ca, Cb = np.eye(nmo*2, dtype=complex), np.eye(nmo*2, dtype=complex)
#    Ua, Ub = _U(fda, fda_), _U(fdb, fdb_)

    taa = np.array(t[0], dtype=complex)
    tab = np.array(t[1], dtype=complex)
    tbb = np.array(t[2], dtype=complex)
    laa = np.array(l[0], dtype=complex)
    lab = np.array(l[1], dtype=complex)
    lbb = np.array(l[2], dtype=complex)
    t, l = (taa, tab, tbb), (laa, lab, lbb)
    d1, d2 = utils.compute_rdm12(t, l)
#    e = utils.compute_energy(d1, d2, eris, time=None)
#    print('check initial energy: {}'.format(e.real+eris.mf.energy_nuc())) 

    d1a_old, d1b_old = utils.build_rdm1(d1)
    d1a_old = utils.compute_phys1(d1a_old, fda, fda_)
    d1b_old = utils.compute_phys1(d1b_old, fdb, fdb_)
    E = np.zeros(N+1,dtype=complex) 
    rdm1a = [] 
    rdm1b = []
    rdm2ab = [] 
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
        if i % every == 0: 
#            d1 = utils.compute_rdm1(t, l)
            d1, d2 = utils.compute_rdm12(t, l)
            d1a_new, d1b_new = utils.build_rdm1(d1) 
            d2ab = utils.build_rdm2ab(d2)
            d1a_new = utils.rotate1(d1a_new, Ca.T.conj())
            d1b_new = utils.rotate1(d1b_new, Cb.T.conj())
            d2ab = utils.rotate2(d2ab, Ca.T.conj(), Cb.T.conj())
#
#            d1a_ = utils.rotate1(d1a_new, Ua)
#            d1b_ = utils.rotate1(d1b_new, Ub)
#            d2ab_ = utils.rotate2(d2ab, Ua, Ub)
#
            d1a_new = utils.compute_phys1(d1a_new, fda, fda_)
            d1b_new = utils.compute_phys1(d1b_new, fdb, fdb_)
#            print('d1a:\n', einsum('pq,ip,jq->ij',d1a_new,ua,ua))
#            print('d1b:\n', einsum('pq,ip,jq->ij',d1b_new,ub,ub))
            d2ab = utils.compute_phys2(d2ab, fda, fdb, fda_, fdb_)
#            print('2*nd: ', eris._eta(d2ab)[0])
#
#            print(np.linalg.norm(d1a_new-d1a_[:nmo,:nmo]))
#            print(np.linalg.norm(d1b_new-d1b_[:nmo,:nmo]))
#            print(np.linalg.norm(d2ab-d2ab_[:nmo,:nmo,:nmo,:nmo]))
#
            rdm1a.append(d1a_new.copy())
            rdm1b.append(d1b_new.copy())
            rdm2ab.append(d2ab.copy())
        if RK == 1 and every == 1:
            # Ehrenfest error
            Fa = utils.compute_phys1(F[0], fda, fda_)
            Fb = utils.compute_phys1(F[1], fdb, fdb_)
            err  = np.linalg.norm((d1a_new-d1a_old)/step-1j*Fa)
            err += np.linalg.norm((d1b_new-d1b_old)/step-1j*Fb)
            print('time: {:.4f}, EE(mH): {}, Xa: {}, Xb: {}, err: {}'.format(
                  time, (E[i] - E[0]).real*1e3, 
                  np.linalg.norm(X[0]), np.linalg.norm(X[1]), err))
            d1a_old = d1a_new.copy()
            d1b_old = d1b_new.copy()
        else:
            print('time: {:.4f}, EE(mH): {}, Xa: {}, Xb: {}'.format(
                  time, (E[i] - E[0]).real*1e3, 
                  np.linalg.norm(X[0]), np.linalg.norm(X[1])))
    rdm1a = np.array(rdm1a, dtype=complex)
    rdm1b = np.array(rdm1b, dtype=complex)
    rdm2ab = np.array(rdm2ab, dtype=complex)
    return (rdm1a, rdm1b), rdm2ab, E

class ERIs_mol:
    def __init__(self, mf, f0=np.zeros(3), w=0.0, td=0.0, 
                 beta=0.0, mu=0.0, picture='I'):
        self.mf = mf
        self.w = w
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.picture = picture
        self.fd = utils.compute_sqrt_fd(mf.mo_energy, beta, mu)

        # integrals in fixed Bogliubov basis
        (fda, fda_), (fdb, fdb_) = self.fd
        h0, h1, eri = utils.mo_ints_mol(mf, f0)[:3]
        h0a_ = utils.make_bogoliubov1(h0[0], fda, fda_)
        h0b_ = utils.make_bogoliubov1(h0[1], fdb, fdb_)
        h1a_ = utils.make_bogoliubov1(h1[0], fda, fda_)
        h1b_ = utils.make_bogoliubov1(h1[1], fdb, fdb_)
        eriaa_ = utils.make_bogoliubov2(eri[0], fda, fda, fda_, fda_)
        eriab_ = utils.make_bogoliubov2(eri[1], fda, fdb, fda_, fdb_)
        eribb_ = utils.make_bogoliubov2(eri[2], fdb, fdb, fdb_, fdb_)
        self.h0_ = h0a_, h0b_
        self.h1_ = h1a_, h1b_
        self.eri_ = eriaa_, eriab_, eribb_

        # integrals in rotating basis
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
        h0 = h1 = eri = None

        if picture == 'I':
            self.Roo = utils.make_Roo(mf.mo_energy, fda , fdb )
            self.Rvv = utils.make_Roo(mf.mo_energy, fda_, fdb_)

    def rotate(self, Ca, Cb, time=None):
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
        noa = nob = self.mf.mol.nao_nr()
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

        if self.picture == 'I':
            self.foo += self.Roo[0]
            self.fOO += self.Roo[1]
            self.fvv += self.Rvv[0]
            self.fVV += self.Rvv[1]
        ha = hb = None

class ERIs_hubbard1D_:
    def __init__(self, model, Pa, Pb, A0=0.0, sigma=1.0, w=0.0, td=0.0, 
                 beta=0.0, mu=0.0, picture='I'):
        self.model = model
        self.P = Pa, Pb
        self.A0 = A0
        self.sigma = sigma
        self.w = w
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.picture = picture
        self.quick = True

        # intgrals in site basis
        V = model.get_umatS()
#        Va = V - V.transpose(0,1,3,2) # should vanish
        h = model.get_tmatS()
        hu, hl = np.triu(h), np.tril(h)

        # diagonalize Fock matrx in site basis
        Fa, Fb = h.copy(), h.copy()
#        Fa += einsum('pqrs,qs->pr',Va,Pa)
        Fa += einsum('pqrs,qs->pr',V ,Pb)
#        Fb += einsum('pqrs,qs->pr',Va,Pb)
        Fb += einsum('pqrs,pr->qs',V ,Pa)
        ea, ua = np.linalg.eigh(Fa)
        eb, ub = np.linalg.eigh(Fb)
        self.mo_energy = ea, eb
        self.mo_coeff = ua, ub # t=0 mo_coeff
        self.fd = utils.compute_sqrt_fd(self.mo_energy, beta, mu)

        # integrals in fixed Bogliubov basis
        (fda, fda_), (fdb, fdb_) = self.fd
        hua, hub = utils.rotate1(hu, ua.T), utils.rotate1(hu, ub.T)
        hla, hlb = utils.rotate1(hl, ua.T), utils.rotate1(hl, ub.T)
#        eriaa = utils.rotate2(Va, ua.T.conj(), ua.T.conj())
#        eribb = utils.rotate2(Va, ub.T.conj(), ub.T.conj())
        eriab = utils.rotate2(V , ua.T.conj(), ub.T.conj())

        hua_ = utils.make_bogoliubov1(hua, fda, fda_)
        hub_ = utils.make_bogoliubov1(hub, fdb, fdb_)
        hla_ = utils.make_bogoliubov1(hla, fda, fda_)
        hlb_ = utils.make_bogoliubov1(hlb, fdb, fdb_)
#        eriaa_ = utils.make_bogoliubov2(eriaa, fda, fda, fda_, fda_)
        self.eriab_ = utils.make_bogoliubov2(eriab, fda, fdb, fda_, fdb_)
#        eribb_ = utils.make_bogoliubov2(eribb, fdb, fdb, fdb_, fdb_)
        self.hu_ = hua_, hub_
        self.hl_ = hla_, hlb_
#        self.eri_ = eriaa_, eriab_, eribb_

        # integrals in rotating basis
        hua = np.array(self.hu_[0],dtype=complex)
        hub = np.array(self.hu_[1],dtype=complex)
        hla = np.array(self.hl_[0],dtype=complex)
        hlb = np.array(self.hl_[1],dtype=complex)
#        eriaa = np.array(self.eri_[0],dtype=complex)
        self.eriab = np.array(self.eriab_,dtype=complex)
#        eribb = np.array(self.eri_[2],dtype=complex)
#        aa = np.zeros((model.L*2,)*4)
        self.hu = hua, hub
        self.hl = hla, hlb
#        self.eri = eriaa, eriab, eribb
#        self.eri = aa, eriab, aa 

        if picture == 'I':
            self.Roo = utils.make_Roo(self.mo_energy, fda , fdb )
            self.Rvv = utils.make_Roo(self.mo_energy, fda_, fdb_)

    def rotate(self, Ca, Cb):
        hua = utils.rotate1(self.hu_[0], Ca)
        hub = utils.rotate1(self.hu_[1], Cb)
        hla = utils.rotate1(self.hl_[0], Ca)
        hlb = utils.rotate1(self.hl_[1], Cb)
#        eriaa = utils.rotate2(self.eri_[0], Ca, Ca)
        self.eriab = utils.rotate2(self.eriab_, Ca, Cb)
#        eribb = utils.rotate2(self.eri_[2], Cb, Cb)
#        aa = np.zeros((self.model.L*2,)*4)
        self.hu = hua, hub
        self.hl = hla, hlb
#        self.eri = eriaa, eriab, eribb
#        self.eri = aa, eriab, aa

    def make_tensors(self, time=None):
        noa = nob = self.model.L
        if time is not None:
            fac = utils.phase_hubbard(self.A0, self.sigma, self.w, self.td, time)
        else:
            fac = 0.0
        ha = self.hu[0] * np.exp(1j*fac) + self.hl[0] * np.exp(-1j*fac) 
        hb = self.hu[1] * np.exp(1j*fac) + self.hl[1] * np.exp(-1j*fac) 

        self.hoo = ha[:noa,:noa].copy()
        self.hvv = ha[noa:,noa:].copy()
        self.hov = ha[:noa,noa:].copy()
#        self.oovv = self.eri[0][:noa,:noa,noa:,noa:].copy()
#        self.oooo = self.eri[0][:noa,:noa,:noa,:noa].copy()
#        self.vvvv = self.eri[0][noa:,noa:,noa:,noa:].copy()
#        self.ovvo = self.eri[0][:noa,noa:,noa:,:noa].copy()
#        self.ovvv = self.eri[0][:noa,noa:,noa:,noa:].copy()
#        self.ooov = self.eri[0][:noa,:noa,:noa,noa:].copy()

        self.hOO = hb[:nob,:nob].copy()
        self.hVV = hb[nob:,nob:].copy()
        self.hOV = hb[:nob,nob:].copy()
#        self.OOVV = self.eri[2][:nob,:nob,nob:,nob:].copy()
#        self.OOOO = self.eri[2][:nob,:nob,:nob,:nob].copy()
#        self.VVVV = self.eri[2][nob:,nob:,nob:,nob:].copy()
#        self.OVVO = self.eri[2][:nob,nob:,nob:,:nob].copy()
#        self.OVVV = self.eri[2][:nob,nob:,nob:,nob:].copy()
#        self.OOOV = self.eri[2][:nob,:nob,:nob,nob:].copy()

        self.oOvV = self.eriab[:noa,:nob,noa:,nob:].copy()
        self.oOoO = self.eriab[:noa,:nob,:noa,:nob].copy()
        self.vVvV = self.eriab[noa:,nob:,noa:,nob:].copy()
        self.oVvO = self.eriab[:noa,nob:,noa:,:nob].copy()
        self.oVoV = self.eriab[:noa,nob:,:noa,nob:].copy()
        self.vOvO = self.eriab[noa:,:nob,noa:,:nob].copy()
        self.oVvV = self.eriab[:noa,nob:,noa:,nob:].copy()
        self.vOvV = self.eriab[noa:,:nob,noa:,nob:].copy()
        self.oOoV = self.eriab[:noa,:nob,:noa,nob:].copy()
        self.oOvO = self.eriab[:noa,:nob,noa:,:nob].copy()

        self.foo, self.fvv = self.hoo.copy(), self.hvv.copy()
        self.fOO, self.fVV = self.hOO.copy(), self.hVV.copy()
#        self.foo += einsum('piqi->pq',self.oooo)
        self.foo += einsum('pIqI->pq',self.oOoO)
#        self.fOO += einsum('piqi->pq',self.OOOO)
        self.fOO += einsum('IpIq->pq',self.oOoO)
#        self.fvv -= einsum('ipqi->pq',self.ovvo)
        self.fvv += einsum('pIqI->pq',self.vOvO)
#        self.fVV -= einsum('ipqi->pq',self.OVVO)
        self.fVV += einsum('IpIq->pq',self.oVoV)

        if self.picture == 'I':
            self.foo += self.Roo[0]
            self.fOO += self.Roo[1]
            self.fvv += self.Rvv[0]
            self.fVV += self.Rvv[1]
        ha = hb = None

    def _eta(self, d2):
        ua, ub = self.mo_coeff
        d2 = utils.rotate2(d2, ua, ub)
        nmo = self.model.L 
        fac = np.zeros((nmo,nmo))
        for i in range(nmo):
            for j in range(nmo):
                fac[i,j] = (-1)**(i+j)
        nd = 2.0*einsum('jjjj',d2)/nmo
        eta = 2.0*einsum('ij,iijj',fac,d2)/nmo
        return nd, eta

class ERIs_hubbard1D:
    def __init__(self, model, Pa, Pb, A0=0.0, sigma=1.0, w=0.0, td=0.0, 
                 beta=0.0, mu=0.0, picture='I'):
        self.model = model
        self.P = Pa, Pb
        self.A0 = A0
        self.sigma = sigma
        self.w = w
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
        self.picture = picture
        self.quick = False

        # intgrals in site basis
        V = model.get_umatS()
        Va = V - V.transpose(0,1,3,2) # should vanish
        h = model.get_tmatS()
        hu, hl = np.triu(h), np.tril(h)

        # diagonalize Fock matrx in site basis
        Fa, Fb = h.copy(), h.copy()
        Fa += einsum('pqrs,qs->pr',Va,Pa)
        Fa += einsum('pqrs,qs->pr',V ,Pb)
        Fb += einsum('pqrs,qs->pr',Va,Pb)
        Fb += einsum('pqrs,pr->qs',V ,Pa)
        ea, ua = np.linalg.eigh(Fa)
        eb, ub = np.linalg.eigh(Fb)
        self.mo_energy = ea, eb
        self.mo_coeff = ua, ub # t=0 mo_coeff
        self.fd = utils.compute_sqrt_fd(self.mo_energy, beta, mu)

        # integrals in fixed Bogliubov basis
        (fda, fda_), (fdb, fdb_) = self.fd
        hua, hub = utils.rotate1(hu, ua.T), utils.rotate1(hu, ub.T)
        hla, hlb = utils.rotate1(hl, ua.T), utils.rotate1(hl, ub.T)
        eriaa = utils.rotate2(Va, ua.T.conj(), ua.T.conj())
        eribb = utils.rotate2(Va, ub.T.conj(), ub.T.conj())
        eriab = utils.rotate2(V , ua.T.conj(), ub.T.conj())

        hua_ = utils.make_bogoliubov1(hua, fda, fda_)
        hub_ = utils.make_bogoliubov1(hub, fdb, fdb_)
        hla_ = utils.make_bogoliubov1(hla, fda, fda_)
        hlb_ = utils.make_bogoliubov1(hlb, fdb, fdb_)
        eriaa_ = utils.make_bogoliubov2(eriaa, fda, fda, fda_, fda_)
        eriab_ = utils.make_bogoliubov2(eriab, fda, fdb, fda_, fdb_)
        eribb_ = utils.make_bogoliubov2(eribb, fdb, fdb, fdb_, fdb_)
        self.hu_ = hua_, hub_
        self.hl_ = hla_, hlb_
        self.eri_ = eriaa_, eriab_, eribb_

        # integrals in rotating basis
        hua = np.array(self.hu_[0],dtype=complex)
        hub = np.array(self.hu_[1],dtype=complex)
        hla = np.array(self.hl_[0],dtype=complex)
        hlb = np.array(self.hl_[1],dtype=complex)
        eriaa = np.array(self.eri_[0],dtype=complex)
        eriab = np.array(self.eri_[1],dtype=complex)
        eribb = np.array(self.eri_[2],dtype=complex)
        self.hu = hua, hub
        self.hl = hla, hlb
        self.eri = eriaa, eriab, eribb
        h0 = h1 = eri = None

        if picture == 'I':
            self.Roo = utils.make_Roo(self.mo_energy, fda , fdb )
            self.Rvv = utils.make_Roo(self.mo_energy, fda_, fdb_)

    def rotate(self, Ca, Cb):
        hua = utils.rotate1(self.hu_[0], Ca)
        hub = utils.rotate1(self.hu_[1], Cb)
        hla = utils.rotate1(self.hl_[0], Ca)
        hlb = utils.rotate1(self.hl_[1], Cb)
        eriaa = utils.rotate2(self.eri_[0], Ca, Ca)
        eriab = utils.rotate2(self.eri_[1], Ca, Cb)
        eribb = utils.rotate2(self.eri_[2], Cb, Cb)
        self.hu = hua, hub
        self.hl = hla, hlb
        self.eri = eriaa, eriab, eribb

    def make_tensors(self, time=None):
        noa = nob = self.model.L
        if time is not None:
            fac = utils.phase_hubbard(self.A0, self.sigma, self.w, self.td, time)
        else:
            fac = 0.0
        ha = self.hu[0] * np.exp(1j*fac) + self.hl[0] * np.exp(-1j*fac) 
        hb = self.hu[1] * np.exp(1j*fac) + self.hl[1] * np.exp(-1j*fac) 

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

        if self.picture == 'I':
            self.foo += self.Roo[0]
            self.fOO += self.Roo[1]
            self.fvv += self.Rvv[0]
            self.fVV += self.Rvv[1]
        ha = hb = None

    def _eta(self, d2):
        ua, ub = self.mo_coeff
        d2 = utils.rotate2(d2, ua, ub)
        nmo = self.model.L 
        fac = np.zeros((nmo,nmo))
        for i in range(nmo):
            for j in range(nmo):
                fac[i,j] = (-1)**(i+j)
        nd = 2.0*einsum('jjjj',d2)/nmo
        eta = 2.0*einsum('ij,iijj',fac,d2)/nmo
        return nd, eta

class ERIs_SIAM:
    def __init__(self, model, Pa=None, Pb=None, mo_energy=None, mo_coeff=None, beta=0.0, mu=0.0, picture='I'):
        self.model = model
        self.P = Pa, Pb
        self.beta = beta
        self.mu = mu # chemical potential
        self.picture = picture
        self.quick = True
        self.L = model.ll + model.lr + 1

        h  = model.get_tmatS()
        h += model.get_vmatS()
        V = model.get_umatS()
        if mo_coeff is None: 
            Fa, Fb = h.copy(), h.copy()
            Fa += einsum('pqrs,qs->pr',V ,Pb)
            Fb += einsum('pqrs,pr->qs',V ,Pa)
            ea, ua = np.linalg.eigh(Fa)
            eb, ub = np.linalg.eigh(Fb)
            mo_energy = ea, eb
            mo_coeff  = ua, ub
        self.mo_energy = mo_energy 
        self.mo_coeff  = mo_coeff # t=0 mo_coeff
        self.fd = utils.compute_sqrt_fd(mo_energy, beta, mu)

        # integrals in fixed Bogliubov basis
        (fda, fda_), (fdb, fdb_) = self.fd
        ua, ub = mo_coeff
        ha, hb = utils.rotate1(h, ua.T), utils.rotate1(h, ub.T)
        eriab = utils.rotate2(V , ua.T.conj(), ub.T.conj())

        self.ha_ = utils.make_bogoliubov1(ha, fda, fda_)
        self.hb_ = utils.make_bogoliubov1(hb, fdb, fdb_)
        self.eriab_ = utils.make_bogoliubov2(eriab, fda, fdb, fda_, fdb_)

        # integrals in rotating basis
        self.ha = np.array(self.ha_,dtype=complex)
        self.hb = np.array(self.hb_,dtype=complex)
        self.eriab = np.array(self.eriab_,dtype=complex)

        if picture == 'I':
            self.Roo = utils.make_Roo(self.mo_energy, fda , fdb )
            self.Rvv = utils.make_Roo(self.mo_energy, fda_, fdb_)

    def rotate(self, Ca, Cb):
        self.ha = utils.rotate1(self.ha_, Ca)
        self.hb = utils.rotate1(self.hb_, Cb)
        self.eriab = utils.rotate2(self.eriab_, Ca, Cb)

    def make_tensors(self, time=None):
        noa = nob = self.L
        self.hoo = self.ha[:noa,:noa].copy()
        self.hvv = self.ha[noa:,noa:].copy()
        self.hov = self.ha[:noa,noa:].copy()
        self.hOO = self.hb[:nob,:nob].copy()
        self.hVV = self.hb[nob:,nob:].copy()
        self.hOV = self.hb[:nob,nob:].copy()
        self.oOvV = self.eriab[:noa,:nob,noa:,nob:].copy()
        self.oOoO = self.eriab[:noa,:nob,:noa,:nob].copy()
        self.vVvV = self.eriab[noa:,nob:,noa:,nob:].copy()
        self.oVvO = self.eriab[:noa,nob:,noa:,:nob].copy()
        self.oVoV = self.eriab[:noa,nob:,:noa,nob:].copy()
        self.vOvO = self.eriab[noa:,:nob,noa:,:nob].copy()
        self.oVvV = self.eriab[:noa,nob:,noa:,nob:].copy()
        self.vOvV = self.eriab[noa:,:nob,noa:,nob:].copy()
        self.oOoV = self.eriab[:noa,:nob,:noa,nob:].copy()
        self.oOvO = self.eriab[:noa,:nob,noa:,:nob].copy()
        self.foo, self.fvv = self.hoo.copy(), self.hvv.copy()
        self.fOO, self.fVV = self.hOO.copy(), self.hVV.copy()
        self.foo += einsum('pIqI->pq',self.oOoO)
        self.fOO += einsum('IpIq->pq',self.oOoO)
        self.fvv += einsum('pIqI->pq',self.vOvO)
        self.fVV += einsum('IpIq->pq',self.oVoV)

        if self.picture == 'I':
            self.foo += self.Roo[0]
            self.fOO += self.Roo[1]
            self.fvv += self.Rvv[0]
            self.fVV += self.Rvv[1]


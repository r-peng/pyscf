import numpy as np
from pyscf import lib, cc
from pyscf.cc.td_roccd_ft import compute_sqrt_fd
from pyscf.cc import td_roccd_utils as utils
import scipy
einsum = lib.einsum

def compute_comm(d1, eris, time):
    eris.make_tensors(time)
    F = np.dot(d1, eris.h)
    return F - F.T.conj()

def compute_energy(d1, eris, time=None):
    eris.make_tensors(time)
    return einsum('pq,qp',eris.h, d1) * 2.0

def update_RK(eris, time, h, RK): # h is step size
    eris.make_tensors(time)
    X1 = 1j*eris.h
    if RK == 1:
        return X1
    if RK == 4:
        eris.make_tensors(time+h*0.5)
        X2 = X3 = 1j*eris.h
        eris.make_tensors(time+h)
        X4 = 1j*eris.h
        X = (X1 + 2.0*X2 + 2.0*X3 + X4)/6.0
        X1 = X2 = X3 = X4 = None
        return X

def kernel(eris, tf, step, RK=1):
    N = int((tf+step*0.1)/step)
    nmo = len(eris.fd)
    d1 = np.diag(eris.fd)
    C = np.eye(nmo, dtype=complex)
    e = compute_energy(d1, eris)
    print('check initial energy: {}'.format(e.real)) 

    E = np.zeros(N+1,dtype=complex) 
    rdm1 = np.zeros((N+1,nmo,nmo),dtype=complex)
    d1_old = d1.copy() 
    for i in range(N+1):
        time = i * step
        F = compute_comm(d1_old, eris, time)
        E[i] = compute_energy(d1_old, eris)
        X = update_RK(eris, time, step, RK)
        C = np.dot(C, scipy.linalg.expm(1j*eris.h*step))
        d1_new = einsum('sr,rp,sq->qp',d1,C,C.conj())
        if RK == 1:
            err = np.linalg.norm((d1_new-d1_old)/step-1j*F)
            print('time: {:.4f}, EE(mH): {}, err: {}'.format(
                  time, (E[i] - E[0]).real*1e3, err))
        else: 
            print('time: {:.4f}, EE(mH): {}'.format(
                  time, (E[i] - E[0]).real*1e3))
        d1_old = d1_new.copy()
        rdm1[i,:,:] = d1_new.copy()
    return rdm1, E
 
class ERIs_1e:
    def __init__(self, h0, h1, f0=np.zeros(3), w=0.0, td=0.0, 
                 beta=0.0, mu=0.0):
        self.w = w
        self.td = td
        self.beta = beta
        self.mu = mu # chemical potential
#        self.picture = picture
        self.fd, _ = compute_sqrt_fd(h0, beta, mu)
        self.fd = np.square(self.fd)
#        self.phys = False
        self.h0 = np.diag(h0)
        self.h1 = h1.copy()

    def make_tensors(self, time=None):
        self.h = self.h0.copy()
        if time is not None: 
            self.h += self.h1 * utils.fac_mol(self.w, self.td, time) 

